from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - depends on local environment
    cv2 = None


SPLITS = ("train", "val", "test")
DATASETS = {
    "spawrious-o2o-easy": Path("SPUME-master/data/spawrious_o2o_easy_metadata.csv"),
    "spawrious-o2o-medium": Path("SPUME-master/data/spawrious_o2o_medium_metadata.csv"),
}


@dataclass(frozen=True)
class Sample:
    dataset: str
    split: str
    class_name: str
    filename: str
    path: Path


class BKTree:
    def __init__(self) -> None:
        self.root: tuple[int, list[int], dict[int, object]] | None = None

    @staticmethod
    def distance(a: int, b: int) -> int:
        return bin(int(a) ^ int(b)).count("1")

    def add(self, value: int, index: int) -> None:
        if self.root is None:
            self.root = (value, [index], {})
            return

        node = self.root
        while True:
            node_value, indices, children = node
            dist = self.distance(value, node_value)
            if dist == 0:
                indices.append(index)
                return
            child = children.get(dist)
            if child is None:
                children[dist] = (value, [index], {})
                return
            node = child  # type: ignore[assignment]

    def query(self, value: int, max_distance: int) -> list[int]:
        if self.root is None:
            return []

        matches: list[int] = []
        stack = [self.root]
        while stack:
            node_value, indices, children = stack.pop()
            dist = self.distance(value, node_value)
            if dist <= max_distance:
                matches.extend(indices)
            low = dist - max_distance
            high = dist + max_distance
            for edge_distance, child in children.items():
                if low <= edge_distance <= high:
                    stack.append(child)  # type: ignore[arg-type]
        return matches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check Spawrious train/val/test filename, SHA256, and perceptual-hash leakage "
            "without modifying training code."
        )
    )
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository/workspace root.")
    parser.add_argument(
        "--dataset",
        choices=[*DATASETS.keys(), "all"],
        default="all",
        help="Dataset to check.",
    )
    parser.add_argument(
        "--hash",
        choices=["sha256", "md5"],
        default="sha256",
        help="Cryptographic hash used for exact image duplicate checks.",
    )
    parser.add_argument(
        "--phash-threshold",
        type=int,
        default=5,
        help="Maximum Hamming distance for perceptual-hash near-duplicate matches.",
    )
    parser.add_argument(
        "--skip-phash",
        action="store_true",
        help="Skip perceptual-hash checks.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, max(1, os.cpu_count() or 1)),
        help="Number of worker processes for image hashing.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path for writing a JSON report.",
    )
    return parser.parse_args()


def load_samples(root: Path, dataset_name: str, metadata_relpath: Path) -> list[Sample]:
    metadata_path = root / metadata_relpath
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    samples: list[Sample] = []
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            split = row["split"].strip()
            if split not in SPLITS:
                continue
            img_path = root.joinpath(*row["img_path"].split("/"))
            filename = row.get("filename") or Path(row["img_path"]).name
            samples.append(
                Sample(
                    dataset=dataset_name,
                    split=split,
                    class_name=row["class_name"].strip(),
                    filename=filename.strip(),
                    path=img_path,
                )
            )
    return samples


def digest_file(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def digest_worker(args: tuple[int, str, str]) -> tuple[int, str | None, str | None]:
    idx, path_text, algorithm = args
    path = Path(path_text)
    if not path.exists():
        return idx, None, "missing"
    try:
        return idx, digest_file(path, algorithm), None
    except Exception as exc:  # pragma: no cover - defensive reporting
        return idx, None, repr(exc)


def phash_file(path: Path) -> int:
    if cv2 is None:
        raise RuntimeError("OpenCV is not available; cannot compute pHash.")
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"OpenCV could not read image: {path}")
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))
    low_freq = dct[:8, :8].copy()
    coeffs = low_freq.flatten()
    median = np.median(coeffs[1:])
    bits = coeffs > median
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def phash_worker(args: tuple[int, str]) -> tuple[int, int | None, str | None]:
    idx, path_text = args
    path = Path(path_text)
    if not path.exists():
        return idx, None, "missing"
    try:
        return idx, phash_file(path), None
    except Exception as exc:  # pragma: no cover - defensive reporting
        return idx, None, repr(exc)


def build_duplicate_counts_by_split(
    samples: list[Sample],
    values: dict[int, str | int],
    splits_to_compare: Iterable[str] = SPLITS,
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    compare_splits = tuple(splits_to_compare)
    value_to_splits: dict[str | int, set[str]] = defaultdict(set)
    for idx, sample in enumerate(samples):
        if sample.split in compare_splits and idx in values:
            value_to_splits[values[idx]].add(sample.split)

    per_split = {split: 0 for split in SPLITS}
    pairwise = {a: {b: 0 for b in SPLITS if b != a} for a in SPLITS}
    for idx, sample in enumerate(samples):
        if sample.split not in compare_splits or idx not in values:
            continue
        other_splits = value_to_splits[values[idx]] - {sample.split}
        if other_splits:
            per_split[sample.split] += 1
            for other in other_splits:
                pairwise[sample.split][other] += 1
    return per_split, pairwise


def split_summary(
    samples: list[Sample],
    missing_indices: set[int],
    filename_cross_split_counts: dict[str, int],
    exact_hash_cross_split_counts: dict[str, int],
    train_test_hash_counts: dict[str, int],
    phash_cross_split_counts: dict[str, int] | None,
    train_test_phash_counts: dict[str, int] | None,
) -> dict[str, dict[str, int | None]]:
    by_split: dict[str, list[tuple[int, Sample]]] = {split: [] for split in SPLITS}
    for idx, sample in enumerate(samples):
        by_split[sample.split].append((idx, sample))

    rows: dict[str, dict[str, int | None]] = {}
    for split in SPLITS:
        split_samples = by_split[split]
        filenames = [sample.filename for _, sample in split_samples]
        rows[split] = {
            "samples": len(split_samples),
            "classes": len({sample.class_name for _, sample in split_samples}),
            "missing_files": sum(1 for idx, _ in split_samples if idx in missing_indices),
            "duplicate_filenames_within_split": len(filenames) - len(set(filenames)),
            "duplicate_filenames_cross_split": filename_cross_split_counts[split],
            "duplicate_exact_hash_cross_split": exact_hash_cross_split_counts[split],
            "duplicate_exact_hash_train_test": train_test_hash_counts[split],
            "near_duplicate_phash_cross_split": (
                None if phash_cross_split_counts is None else phash_cross_split_counts[split]
            ),
            "near_duplicate_phash_train_test": (
                None if train_test_phash_counts is None else train_test_phash_counts[split]
            ),
        }
    return rows


def filename_checks(samples: list[Sample]) -> tuple[dict[str, int], dict[str, dict[str, int]], list[dict[str, object]]]:
    filename_values = {idx: sample.filename for idx, sample in enumerate(samples)}
    per_split, pairwise = build_duplicate_counts_by_split(samples, filename_values)

    filename_to_samples: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        filename_to_samples[sample.filename].append(sample)

    examples: list[dict[str, object]] = []
    for filename, grouped in filename_to_samples.items():
        splits = sorted({sample.split for sample in grouped})
        if len(splits) > 1:
            examples.append(
                {
                    "filename": filename,
                    "splits": splits,
                    "count": len(grouped),
                    "paths": [str(sample.path) for sample in grouped[:6]],
                }
            )
        if len(examples) >= 10:
            break

    return per_split, pairwise, examples


def exact_hash_checks(
    samples: list[Sample],
    algorithm: str,
    workers: int,
) -> tuple[dict[int, str], set[int], list[dict[str, object]]]:
    hashes: dict[int, str] = {}
    missing: set[int] = set()
    errors: list[dict[str, object]] = []

    total = len(samples)
    tasks = [(idx, str(sample.path), algorithm) for idx, sample in enumerate(samples)]
    executor = None
    if workers <= 1:
        iterator = map(digest_worker, tasks)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        iterator = executor.map(digest_worker, tasks, chunksize=128)

    try:
        for completed, (idx, digest, error) in enumerate(iterator, start=1):
            if digest is not None:
                hashes[idx] = digest
            else:
                missing.add(idx)
                errors.append({"path": str(samples[idx].path), "error": error})
            if completed % 10000 == 0:
                print(f"  hashed {completed}/{total} images with {algorithm}...", file=sys.stderr)
    finally:
        if executor is not None:
            executor.shutdown()

    return hashes, missing, errors


def train_test_only_counts(
    samples: list[Sample],
    values: dict[int, str | int],
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    return build_duplicate_counts_by_split(samples, values, splits_to_compare=("train", "test"))


def sample_duplicate_examples(
    samples: list[Sample],
    values: dict[int, str | int],
    split_a: str,
    split_b: str,
    limit: int = 10,
) -> list[dict[str, object]]:
    grouped: dict[str | int, dict[str, list[Sample]]] = defaultdict(lambda: defaultdict(list))
    for idx, sample in enumerate(samples):
        if sample.split in {split_a, split_b} and idx in values:
            grouped[values[idx]][sample.split].append(sample)

    examples: list[dict[str, object]] = []
    for value, by_split in grouped.items():
        if split_a in by_split and split_b in by_split:
            examples.append(
                {
                    "value": str(value),
                    split_a: [str(sample.path) for sample in by_split[split_a][:3]],
                    split_b: [str(sample.path) for sample in by_split[split_b][:3]],
                }
            )
        if len(examples) >= limit:
            break
    return examples


def phash_checks(
    samples: list[Sample],
    threshold: int,
    workers: int,
) -> tuple[dict[int, int], dict[str, int], dict[str, int], list[dict[str, object]]]:
    if cv2 is None:
        raise RuntimeError("OpenCV is not available; run with --skip-phash or install opencv-python.")

    phashes: dict[int, int] = {}
    total = len(samples)
    tasks = [(idx, str(sample.path)) for idx, sample in enumerate(samples)]
    executor = None
    if workers <= 1:
        iterator = map(phash_worker, tasks)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        iterator = executor.map(phash_worker, tasks, chunksize=128)

    try:
        for completed, (idx, value, error) in enumerate(iterator, start=1):
            if value is not None:
                phashes[idx] = value
            elif error != "missing":
                print(f"  pHash error for {samples[idx].path}: {error}", file=sys.stderr)
            if completed % 10000 == 0:
                print(f"  computed pHash {completed}/{total} images...", file=sys.stderr)
    finally:
        if executor is not None:
            executor.shutdown()

    cross_match_indices = {split: set() for split in SPLITS}
    train_test_match_indices = {split: set() for split in SPLITS}
    examples: list[dict[str, object]] = []

    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        right_indices = [idx for idx, sample in enumerate(samples) if sample.split == right and idx in phashes]
        tree = BKTree()
        for idx in right_indices:
            tree.add(phashes[idx], idx)

        for left_idx, sample in enumerate(samples):
            if sample.split != left or left_idx not in phashes:
                continue
            matches = tree.query(phashes[left_idx], threshold)
            if not matches:
                continue
            cross_match_indices[left].add(left_idx)
            for match_idx in matches:
                cross_match_indices[right].add(match_idx)

            if {left, right} == {"train", "test"}:
                train_test_match_indices[left].add(left_idx)
                train_test_match_indices[right].update(matches)
                if len(examples) < 10:
                    first_match = matches[0]
                    examples.append(
                        {
                            "hamming_distance": BKTree.distance(phashes[left_idx], phashes[first_match]),
                            left: str(sample.path),
                            right: str(samples[first_match].path),
                        }
                    )

    cross_counts = {split: len(indices) for split, indices in cross_match_indices.items()}
    train_test_counts = {split: len(indices) for split, indices in train_test_match_indices.items()}
    return phashes, cross_counts, train_test_counts, examples


def print_table(title: str, rows: dict[str, dict[str, int | None]]) -> None:
    columns = [
        "split",
        "samples",
        "classes",
        "missing_files",
        "duplicate_filenames_cross_split",
        "duplicate_exact_hash_cross_split",
        "duplicate_exact_hash_train_test",
        "near_duplicate_phash_train_test",
    ]
    print(f"\n{title}")
    print("| " + " | ".join(columns) + " |")
    print("|" + "|".join(["---"] * len(columns)) + "|")
    for split in SPLITS:
        row = rows[split]
        values = [split] + [row[column] for column in columns[1:]]
        print("| " + " | ".join("NA" if value is None else str(value) for value in values) + " |")


def analyze_dataset(root: Path, dataset_name: str, metadata_relpath: Path, args: argparse.Namespace) -> dict[str, object]:
    print(f"\n=== Checking {dataset_name} ===")
    samples = load_samples(root, dataset_name, metadata_relpath)
    print(f"loaded {len(samples)} metadata rows")

    filename_counts, filename_pairwise, filename_examples = filename_checks(samples)

    hashes, missing_indices, hash_errors = exact_hash_checks(samples, args.hash, args.workers)
    exact_counts, exact_pairwise = build_duplicate_counts_by_split(samples, hashes)
    train_test_hash_counts, train_test_hash_pairwise = train_test_only_counts(samples, hashes)
    hash_examples = sample_duplicate_examples(samples, hashes, "train", "test")

    phash_counts = None
    train_test_phash_counts = None
    phash_examples: list[dict[str, object]] = []
    if args.skip_phash:
        phash_error = "skipped"
    else:
        try:
            _, phash_counts, train_test_phash_counts, phash_examples = phash_checks(
                samples,
                args.phash_threshold,
                args.workers,
            )
            phash_error = None
        except Exception as exc:
            phash_error = repr(exc)

    rows = split_summary(
        samples=samples,
        missing_indices=missing_indices,
        filename_cross_split_counts=filename_counts,
        exact_hash_cross_split_counts=exact_counts,
        train_test_hash_counts=train_test_hash_counts,
        phash_cross_split_counts=phash_counts,
        train_test_phash_counts=train_test_phash_counts,
    )
    print_table(f"{dataset_name} split summary", rows)

    print("\nPairwise duplicate filename counts (rows counted in row split):")
    print(json.dumps(filename_pairwise, ensure_ascii=False, indent=2))
    print(f"\nPairwise duplicate {args.hash} counts (rows counted in row split):")
    print(json.dumps(exact_pairwise, ensure_ascii=False, indent=2))
    print(f"\nTrain/test duplicate {args.hash} counts:")
    print(json.dumps(train_test_hash_pairwise, ensure_ascii=False, indent=2))
    if phash_counts is not None:
        print(f"\nTrain/test near-duplicate pHash examples, threshold <= {args.phash_threshold}:")
        print(json.dumps(phash_examples, ensure_ascii=False, indent=2))
    elif phash_error:
        print(f"\npHash check: {phash_error}")

    report = {
        "dataset": dataset_name,
        "metadata": str(root / metadata_relpath),
        "hash_algorithm": args.hash,
        "phash_threshold": args.phash_threshold,
        "split_summary": rows,
        "pairwise_duplicate_filenames": filename_pairwise,
        "pairwise_duplicate_exact_hash": exact_pairwise,
        "train_test_duplicate_exact_hash": train_test_hash_pairwise,
        "filename_duplicate_examples": filename_examples,
        "train_test_exact_hash_examples": hash_examples,
        "train_test_phash_examples": phash_examples,
        "hash_errors": hash_errors[:20],
        "phash_error": phash_error,
    }
    return report


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    dataset_names = DATASETS.keys() if args.dataset == "all" else [args.dataset]

    reports = []
    for dataset_name in dataset_names:
        reports.append(analyze_dataset(root, dataset_name, DATASETS[dataset_name], args))

    if args.json_out:
        output_path = args.json_out
        if not output_path.is_absolute():
            output_path = root / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote JSON report to {output_path}")


if __name__ == "__main__":
    main()
