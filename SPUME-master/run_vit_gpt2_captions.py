import argparse

from extract_concepts import get_data_folder
from img_captioning import VITGPT2_CAPTIONING


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="waterbirds")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--model_dir", default=r"d:/SPUME/vit-gpt2-model")
    args = parser.parse_args()

    img_path, csv_path = get_data_folder(args.dataset)
    model = VITGPT2_CAPTIONING(local_dir=args.model_dir)
    out_path = model.get_img_captions(
        img_path,
        csv_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(out_path)


if __name__ == "__main__":
    main()
