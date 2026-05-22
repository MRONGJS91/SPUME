import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_RATIOS = (0.8, 0.1, 0.1)


def is_image_file(path):
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def resolve_dataset_root(root):
    if not root.exists():
        raise FileNotFoundError(f"路径不存在: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"给定路径不是目录: {root}")

    child_dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    if not child_dirs:
        raise ValueError(f"目录为空，未找到任何子目录: {root}")

    if all(p.name.isdigit() for p in child_dirs):
        return root

    if len(child_dirs) == 1:
        nested = child_dirs[0]
        nested_dirs = [p for p in sorted(nested.iterdir()) if p.is_dir()]
        if nested_dirs and all(p.name.isdigit() for p in nested_dirs):
            return nested

    names = ", ".join(p.name for p in child_dirs[:10])
    raise ValueError(
        "目录结构不符合预期。期望是 `root/0/...` 或 `root/<dataset_dir>/0/...`。"
        f" 当前发现的一级子目录有: {names}"
    )


def assign_split(index, total):
    train_end = int(total * SPLIT_RATIOS[0])
    val_end = train_end + int(total * SPLIT_RATIOS[1])
    if index < train_end:
        return "train"
    if index < val_end:
        return "val"
    return "test"


def build_rows(dataset_root, project_root):
    env_dirs = [p for p in sorted(dataset_root.iterdir()) if p.is_dir()]
    if not env_dirs:
        raise ValueError(f"未找到环境目录: {dataset_root}")

    class_names = sorted(
        {
            class_dir.name
            for env_dir in env_dirs
            for bg_dir in sorted(env_dir.iterdir())
            if bg_dir.is_dir()
            for class_dir in sorted(bg_dir.iterdir())
            if class_dir.is_dir()
        }
    )
    if not class_names:
        raise ValueError(f"未找到类别目录: {dataset_root}")
    class_to_y = {name: idx for idx, name in enumerate(class_names)}

    rows = []
    split_counter = Counter()
    class_counter = Counter()
    split_class_counter = Counter()

    for env_dir in env_dirs:
        bg_dirs = [p for p in sorted(env_dir.iterdir()) if p.is_dir()]
        if not bg_dirs:
            raise ValueError(f"环境目录下没有背景目录: {env_dir}")

        for bg_dir in bg_dirs:
            class_dirs = [p for p in sorted(bg_dir.iterdir()) if p.is_dir()]
            if not class_dirs:
                raise ValueError(f"背景目录下没有类别目录: {bg_dir}")

            for class_dir in class_dirs:
                image_paths = [p for p in sorted(class_dir.iterdir()) if is_image_file(p)]
                if not image_paths:
                    raise ValueError(f"类别目录下没有图片: {class_dir}")

                total = len(image_paths)
                env_name = f"env_{env_dir.name}_{bg_dir.name}"
                class_name = class_dir.name
                y = class_to_y[class_name]

                for index, image_path in enumerate(image_paths):
                    split = assign_split(index, total)
                    try:
                        img_path = image_path.relative_to(project_root).as_posix()
                    except ValueError:
                        img_path = str(image_path.resolve())

                    row = {
                        "img_path": img_path,
                        "class_name": class_name,
                        "y": y,
                        "split": split,
                        "env": env_name,
                        "filename": image_path.name,
                    }
                    rows.append(row)
                    split_counter[split] += 1
                    class_counter[class_name] += 1
                    split_class_counter[(split, class_name)] += 1

    return rows, class_to_y, split_counter, class_counter, split_class_counter


def print_summary(class_to_y, split_counter, class_counter, split_class_counter):
    print("== 类别映射 ==")
    for class_name in sorted(class_to_y):
        print(f"- {class_name}: {class_to_y[class_name]}")
    print()

    print("== 每个 split 的样本数 ==")
    for split in ["train", "val", "test"]:
        print(f"- {split}: {split_counter[split]}")
    print()

    print("== 每个类别的样本数 ==")
    for class_name in sorted(class_counter):
        print(f"- {class_name}: {class_counter[class_name]}")
    print()

    print("== 每个 split / 类别 的样本数 ==")
    grouped = defaultdict(list)
    for (split, class_name), count in split_class_counter.items():
        grouped[split].append((class_name, count))
    for split in ["train", "val", "test"]:
        print(f"[{split}]")
        for class_name, count in sorted(grouped.get(split, [])):
            print(f"  - {class_name}: {count}")
        print()


def write_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["img_path", "class_name", "y", "split", "env", "filename"]
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Build metadata CSV for Spawrious o2o_easy."
    )
    parser.add_argument(
        "--root",
        default=r"D:\SPUME\spawrious224__o2o_easy",
        help="Spawrious dataset root path",
    )
    parser.add_argument(
        "--output",
        default=r"D:\SPUME\SPUME-master\data\spawrious_o2o_easy_metadata.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    root = Path(args.root)
    output_path = Path(args.output)

    dataset_root = resolve_dataset_root(root)
    rows, class_to_y, split_counter, class_counter, split_class_counter = build_rows(
        dataset_root, project_root
    )
    write_csv(rows, output_path)

    print(f"Resolved dataset root: {dataset_root}")
    print(f"Project root: {project_root}")
    print(f"Output CSV: {output_path}")
    print(f"Total rows: {len(rows)}")
    print()
    print_summary(class_to_y, split_counter, class_counter, split_class_counter)


if __name__ == "__main__":
    main()
