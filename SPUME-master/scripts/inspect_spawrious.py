import argparse
import random
import sys
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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

    # Spawrious 常见打包形式:
    # 1. root/0, root/1, ...
    # 2. root/spawrious224/0, root/spawrious224/1, ...
    if all(p.name.isdigit() for p in child_dirs):
        return root, None

    if len(child_dirs) == 1:
        nested = child_dirs[0]
        nested_dirs = [p for p in sorted(nested.iterdir()) if p.is_dir()]
        if nested_dirs and all(p.name.isdigit() for p in nested_dirs):
            return nested, (
                f"检测到外层包装目录 `{root.name}`，实际数据目录为 `{nested.name}`。"
            )

    names = ", ".join(p.name for p in child_dirs[:10])
    raise ValueError(
        "目录结构不符合预期。期望是 `root/0/...` 或 `root/<dataset_dir>/0/...`。"
        f" 当前发现的一级子目录有: {names}"
    )


def collect_level_names(root):
    return [p.name for p in sorted(root.iterdir())]


def inspect_env_dir(env_dir, sample_limit):
    class_dirs = [p for p in sorted(env_dir.iterdir()) if p.is_dir()]
    if not class_dirs:
        raise ValueError(f"环境目录下没有类别目录: {env_dir}")

    class_summaries = []
    sample_paths = []
    for class_dir in class_dirs:
        images = [p for p in sorted(class_dir.rglob("*")) if is_image_file(p)]
        class_summaries.append((class_dir.name, len(images)))
        for image_path in images[: min(3, len(images))]:
            sample_paths.append(image_path)

    if not any(count > 0 for _, count in class_summaries):
        raise ValueError(f"未在环境目录下找到图片文件: {env_dir}")

    if len(sample_paths) > sample_limit:
        sample_paths = random.sample(sample_paths, sample_limit)
        sample_paths.sort()

    return class_summaries, sample_paths


def main():
    parser = argparse.ArgumentParser(
        description="Inspect Spawrious directory structure without relying on training code."
    )
    parser.add_argument(
        "--root",
        default=r"D:\SPUME\spawrious224__o2o_easy",
        help="Spawrious dataset root path",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=10,
        help="Maximum number of sampled image paths to print",
    )
    args = parser.parse_args()

    root = Path(args.root)

    try:
        dataset_root, note = resolve_dataset_root(root)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    print("=== Spawrious Directory Inspection ===")
    print(f"Input root: {root}")
    print(f"Resolved dataset root: {dataset_root}")
    if note:
        print(f"Note: {note}")
    print()

    print("== 一级目录结构 ==")
    for name in collect_level_names(root):
        print(f"- {name}")
    print()

    print("== 数据目录下的 split / env ==")
    env_dirs = [p for p in sorted(dataset_root.iterdir()) if p.is_dir()]
    for env_dir in env_dirs:
        print(f"- {env_dir.name}")
    print()

    total_classes = 0
    total_images = 0
    all_samples = []

    print("== 每个 split / env 的类别统计 ==")
    for env_dir in env_dirs:
        try:
            class_summaries, sample_paths = inspect_env_dir(env_dir, args.sample_limit)
        except Exception as exc:
            print(f"[ERROR] {exc}")
            sys.exit(1)

        total_classes += len(class_summaries)
        env_image_count = sum(count for _, count in class_summaries)
        total_images += env_image_count
        all_samples.extend(sample_paths)

        print(f"[{env_dir.name}] 类别目录数: {len(class_summaries)}, 图片总数: {env_image_count}")
        for class_name, image_count in class_summaries:
            print(f"  - {class_name}: {image_count}")
        print()

    if len(all_samples) > args.sample_limit:
        all_samples = random.sample(all_samples, args.sample_limit)
        all_samples.sort()

    print("== 抽样图片相对路径 ==")
    for sample_path in all_samples:
        print(f"- {sample_path.relative_to(dataset_root)}")
    print()

    print("== 汇总 ==")
    print(f"环境目录数: {len(env_dirs)}")
    print(f"类别目录总数: {total_classes}")
    print(f"图片总数: {total_images}")


if __name__ == "__main__":
    main()
