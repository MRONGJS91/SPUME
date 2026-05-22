from transformers import (
    VisionEncoderDecoderModel,
    ViTFeatureExtractor,
    AutoTokenizer,
    BlipProcessor,
    BlipForConditionalGeneration,
)
import torch
from PIL import Image, ImageFile
import os
from tqdm import tqdm
import utils
import argparse
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import csv
from torch.amp import autocast


ImageFile.LOAD_TRUNCATED_IMAGES = True


def normalize_image_key(path):
    """Keep image identifiers as stable relative-path-like keys."""
    key = str(path).strip().replace("\\", "/")
    while key.startswith("./"):
        key = key[2:]
    return key


def find_hf_snapshot_dir(repo_id):
    """Return a local Hugging Face snapshot path if the model has been cached."""
    cache_root = os.path.join(
        os.environ.get("USERPROFILE", os.path.expanduser("~")),
        ".cache",
        "huggingface",
        "hub",
        f"models--{repo_id.replace('/', '--')}",
        "snapshots",
    )
    if not os.path.isdir(cache_root):
        return None
    candidates = [
        os.path.join(cache_root, name)
        for name in os.listdir(cache_root)
        if os.path.isdir(os.path.join(cache_root, name))
    ]
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def collate_fn(data):
    """Transform a list of tuples into a batch

    Args:
        data (list[tuple[np.ndarray, int, np.ndarray, np.ndarray]]): a list of tuples sampled from a dataset

    Returns:
        tensor: a list of tensor data
    """
    # data = a list of tuples
    batch_size = len(data)
    batch1 = [data[i][0] for i in range(batch_size)]
    batch2 = [data[i][1] for i in range(batch_size)]
    batch3 = [data[i][2] for i in range(batch_size)]
    return batch1, batch2, batch3


class ImageData(Dataset):
    def __init__(self, img_folder, meta_data):
        """Initialize the dataset

        Args:
            img_folder (str): folder containing images
            meta_data (pd.DataFrame): metadata of the dataset
        """
        self.meta_data = meta_data
        self.img_folder = img_folder

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        """Return an image, its label, and its name

        Args:
            idx (int): index of the image

        Returns:
            (PIL.Image.Image, int, str): an image, its label, and its name
        """
        path_col = "img_filename" if "img_filename" in self.meta_data.columns else "img_path"
        image_path = self.meta_data.iloc[idx][path_col]
        image_path = self._resolve_image_path(image_path)
        with Image.open(image_path) as temp:
            if temp.mode != "RGB":
                temp = temp.convert(mode="RGB")
            image = temp.copy()
        image_name = normalize_image_key(self.meta_data.iloc[idx][path_col])
        label = self.meta_data.iloc[idx]["y"]
        return image, label, image_name

    def _resolve_image_path(self, image_path):
        candidates = [
            image_path,
            os.path.join(self.img_folder, image_path),
            os.path.join(os.path.dirname(self.img_folder), image_path),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(
            f"Image file not found for {image_path}. Tried: {candidates}"
        )


def filter_caption_metadata(metadata_df):
    if "split" not in metadata_df.columns:
        return metadata_df
    split_series = metadata_df["split"]
    if split_series.dtype.kind in {"i", "u", "f"}:
        return metadata_df[split_series != 2]
    return metadata_df[split_series.astype(str).str.lower() != "test"]


def get_path_column(metadata_df):
    if "img_filename" in metadata_df.columns:
        return "img_filename"
    if "img_path" in metadata_df.columns:
        return "img_path"
    raise ValueError("metadata must contain either 'img_filename' or 'img_path'")


class VITGPT2_CAPTIONING:
    def __init__(self, max_length=16, num_beams=4, local_dir=None):
        """Initialize the ViT-GPT2 model for image captioning

        Args:
            max_length (int, optional): Maximum description length. Defaults to 16.
            num_beams (int, optional): Number of beams used in beam search. Defaults to 4.
            local_dir (str, optional): local directory for a pretrained vit-gpt2 model.
        """
        default_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "vit-gpt2-model")
        )
        local_dir = local_dir or default_dir
        if os.path.isdir(local_dir) and os.listdir(local_dir):
            pretrained_name = local_dir
            print(f"Using local vit-gpt2 model from {local_dir}")
        else:
            pretrained_name = "nlpconnect/vit-gpt2-image-captioning"
            print(
                f"Local vit-gpt2 model not found at {local_dir}. "
                "Attempting to download from Hugging Face."
            )

        self.model = VisionEncoderDecoderModel.from_pretrained(pretrained_name)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(pretrained_name)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        gpu = ",".join([str(i) for i in utils.get_free_gpu()[0:1]])
        utils.set_gpu(gpu)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    @staticmethod
    def _is_valid_caption_file(save_path, expected_rows):
        """Check whether an existing caption csv can be reused safely."""
        if not os.path.exists(save_path):
            return False
        if os.path.getsize(save_path) == 0:
            return False

        with open(save_path, "rb") as f:
            head = f.read(4096)
            if b"\x00" in head:
                return False

        try:
            with open(save_path, "r", newline="", encoding="utf-8") as f:
                row_count = sum(1 for _ in csv.reader(f))
        except Exception:
            return False
        return row_count == expected_rows

    @staticmethod
    def _read_existing_caption_rows(save_path):
        """Load existing caption rows for resumable generation."""
        if not os.path.exists(save_path) or os.path.getsize(save_path) == 0:
            return []
        try:
            with open(save_path, "r", newline="", encoding="utf-8") as f:
                rows = [row for row in csv.reader(f) if len(row) >= 3]
        except Exception:
            return []
        return rows

    def predict_step(self, data, names):
        """Generate image captions for a batch of images

        Args:
            data (list[PIL.Image.Image]): a list of Image objects
            names (list[str]): a list of image names

        Returns:
            list[str]: a list of image captions in the format of "image_name,caption"
        """
        with torch.no_grad():
            pixel_values = self.feature_extractor(
                images=data, return_tensors="pt"
            ).pixel_values
            pixel_values = pixel_values.to(self.device)
            output_ids = self.model.generate(pixel_values, **self.gen_kwargs)
            preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            msgs = []
            for i in range(len(preds)):
                msgs.append(f"{names[i]},{preds[i].strip()}")
        return msgs

    def get_img_captions(
        self,
        img_folder,
        csv_path,
        batch_size=256,
        num_workers=0,
        max_samples=0,
        log_every=1,
    ):
        """Generate image captions for images specified in the csv file

        Args:
            img_folder (str): folder containing images
            csv_path (str): metadata csv file
            batch_size (int, optional): Batch size. Defaults to 256.

        Raises:
            ValueError: if the csv file does not exist

        Returns:
            str: path to the generated csv file containing image captions
        """
        if not os.path.exists(csv_path):
            raise ValueError(f"{csv_path} does not exist")
        metadata_df = pd.read_csv(csv_path)
        metadata_df = filter_caption_metadata(metadata_df)
        path_col = get_path_column(metadata_df)
        if max_samples and max_samples > 0:
            metadata_df = metadata_df.head(max_samples).reset_index(drop=True)

        save_name = "vit-gpt2_captions.csv"
        if max_samples and max_samples > 0:
            save_name = f"vit-gpt2_first{max_samples}_captions.csv"
        save_path = os.path.join(img_folder, save_name)
        if self._is_valid_caption_file(save_path, len(metadata_df)):
            print(f"{save_path} has been generated")
            return save_path

        existing_rows = self._read_existing_caption_rows(save_path)
        done_names = {row[0] for row in existing_rows}
        if existing_rows:
            metadata_df = metadata_df[~metadata_df[path_col].isin(done_names)]
            metadata_df = metadata_df.reset_index(drop=True)
            print(
                f"Resuming captioning: {len(done_names)}/{len(done_names) + len(metadata_df)} done"
            )

        dataset = ImageData(img_folder, metadata_df)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn
        )
        count = len(done_names)
        total = count + len(metadata_df)
        timer = utils.Timer()
        open_mode = "a" if existing_rows else "w"
        with open(save_path, open_mode, newline="", encoding="utf-8") as fout:
            writer = csv.writer(fout)
            for batch_idx, (data, labels, names) in enumerate(dataloader):
                msgs = self.predict_step(data, names)
                for i in range(len(msgs)):
                    img_name, caption = msgs[i].split(",", 1)
                    writer.writerow([img_name, caption, labels[i]])
                fout.flush()
                count += len(data)
                elapsed_time = timer.t()
                est_time = elapsed_time / max(count - len(done_names), 1) * len(metadata_df)
                if batch_idx % max(log_every, 1) == 0:
                    example_name, example_caption = msgs[0].split(",", 1)
                    print(
                        f"[vit-gpt2] Progress {count}/{total} ({count / total * 100:.2f}%) | "
                        f"elapsed={utils.time_str(elapsed_time)} | est_total={utils.time_str(est_time)}"
                    )
                    print(f"[vit-gpt2] Current image: {example_name}")
                    print(f"[vit-gpt2] Caption example: {example_caption}")
        return save_path


class BLIP_CAPTIONING:
    def __init__(self, max_length=16, num_beams=4, local_dir=None):
        """Initliaze the BLIP model for image captioning

        Args:
            max_length (int, optional): Maximum description length. Defaults to 16.
            num_beams (int, optional): Number of beams used in beam search. Defaults to 4.
            local_dir (str, optional): local directory for a pretrained BLIP model.
        """
        default_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "blip-image-captioning-large")
        )
        local_dir = local_dir or default_dir
        cached_snapshot = find_hf_snapshot_dir("Salesforce/blip-image-captioning-large")
        if os.path.isdir(local_dir) and os.listdir(local_dir):
            pretrained_name = local_dir
            print(f"Using local BLIP model from {local_dir}")
        elif cached_snapshot is not None:
            pretrained_name = cached_snapshot
            print(f"Using cached BLIP model from {cached_snapshot}")
        else:
            pretrained_name = "Salesforce/blip-image-captioning-large"
            print(
                f"Local BLIP model not found at {local_dir}. "
                "Attempting to download from Hugging Face."
            )

        self.processor = BlipProcessor.from_pretrained(pretrained_name)
        self.model = BlipForConditionalGeneration.from_pretrained(pretrained_name)

        gpu = ",".join([str(i) for i in utils.get_free_gpu()[0:1]])
        utils.set_gpu(gpu)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.use_half = self.device.type == "cuda"
        if self.use_half:
            self.model.half()
        self.model.eval()
        self.gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    def predict_step(self, data, names):
        """Generate image captions for a batch of images

        Args:
            data (list[PIL.Image.Image]): a list of Image objects
            names (list[str]): a list of image names

        Returns:
            list[str]: a list of image captions in the format of "image_name,caption"
        """
        with torch.no_grad():
            inputs = self.processor(images=data, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with autocast(device_type="cuda", enabled=self.use_half):
                output_ids = self.model.generate(**inputs, **self.gen_kwargs)
            preds = self.processor.batch_decode(output_ids, skip_special_tokens=True)

            msgs = []
            for i in range(len(preds)):
                msgs.append(f"{names[i]},{preds[i].strip()}")
        return msgs

    def get_img_captions(
        self,
        img_folder,
        csv_path,
        batch_size=8,
        num_workers=0,
        max_samples=0,
        log_every=1,
    ):
        """Generate image captions for images specified in the csv file

        Args:
            img_folder (str): folder containing images
            csv_path (str): metadata csv file
            batch_size (int, optional): Batch size.

        Raises:
            ValueError: if the csv file does not exist

        Returns:
            str: path to the generated csv file containing image captions
        """
        if not os.path.exists(csv_path):
            raise ValueError(f"{csv_path} does not exist")

        metadata_df = pd.read_csv(csv_path)
        metadata_df = filter_caption_metadata(metadata_df)
        path_col = get_path_column(metadata_df)
        if max_samples and max_samples > 0:
            metadata_df = metadata_df.head(max_samples).reset_index(drop=True)

        save_name = "blip_captions.csv"
        if max_samples and max_samples > 0:
            save_name = f"blip_first{max_samples}_captions.csv"
        save_path = os.path.join(img_folder, save_name)
        if VITGPT2_CAPTIONING._is_valid_caption_file(save_path, len(metadata_df)):
            print(f"{save_path} has been generated")
            return save_path
        existing_rows = VITGPT2_CAPTIONING._read_existing_caption_rows(save_path)
        done_names = {row[0] for row in existing_rows}
        if existing_rows:
            metadata_df = metadata_df[~metadata_df[path_col].isin(done_names)]
            metadata_df = metadata_df.reset_index(drop=True)
            print(
                f"Resuming captioning: {len(done_names)}/{len(done_names) + len(metadata_df)} done"
            )
        dataset = ImageData(img_folder, metadata_df)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn
        )
        timer = utils.Timer()
        count = len(done_names)
        total = count + len(metadata_df)
        open_mode = "a" if existing_rows else "w"
        with open(save_path, open_mode, newline="", encoding="utf-8") as fout:
            writer = csv.writer(fout)
            for batch_idx, (data, labels, names) in enumerate(dataloader):
                msgs = self.predict_step(data, names)
                for i in range(len(msgs)):
                    img_name, caption = msgs[i].split(",", 1)
                    writer.writerow([img_name, caption, labels[i]])
                fout.flush()
                count += len(data)
                elapsed_time = timer.t()
                est_time = elapsed_time / max(count - len(done_names), 1) * len(metadata_df)
                if batch_idx % max(log_every, 1) == 0:
                    example_name, example_caption = msgs[0].split(",", 1)
                    print(
                        f"[blip] Progress {count}/{total} ({count / total * 100:.2f}%) | "
                        f"elapsed={utils.time_str(elapsed_time)} | est_total={utils.time_str(est_time)}"
                    )
                    print(f"[blip] Current image: {example_name}")
                    print(f"[blip] Caption example: {example_caption}")
        return save_path
