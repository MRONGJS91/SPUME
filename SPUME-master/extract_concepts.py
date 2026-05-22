import spacy
from collections import Counter
from spacy.tokenizer import Tokenizer
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd
import utils
from img_captioning import VITGPT2_CAPTIONING, BLIP_CAPTIONING
import argparse
import csv


def get_caption_tag(path):
    file_name = os.path.basename(path)
    if file_name.endswith("_captions.csv"):
        return file_name[: -len("_captions.csv")]
    return os.path.splitext(file_name)[0]


def normalize_image_key(path):
    """Normalize an image identifier without collapsing it to basename."""
    key = str(path).strip().replace("\\", "/")
    while key.startswith("./"):
        key = key[2:]
    return key


def to_singular(nlp, text):
    """
    Convert a plural noun to singular form
    """
    doc = nlp(text)
    if len(doc) == 1:
        return doc[0].lemma_
    else:
        return doc[:-1].text + doc[-2].whitespace_ + doc[-1].lemma_


def get_adj_pairs(doc):
    """
    从文档的名词短语中提取形容词
    
    遍历文档中的所有名词短语(noun chunks),识别其中的形容词(ADJ),
    并将形容词以"文本:adj"的格式收集到集合中,最后返回去重后的形容词列表。
    
    Args:
        doc: spaCy的Doc对象,包含已进行语法分析的文本
        
    Returns:
        list: 包含唯一形容词的列表,每个形容词格式为"{text}:adj"
              例如: ["beautiful:adj", "large:adj"]
    """
    adj_set = set()
    for chunk in doc.noun_chunks:
        adj = []
        split = False
        noun = ""
        # 遍历名词短语中的每个token,提取形容词
        for tok in chunk:
            if tok.pos_ == "ADJ":
                adj.append(f"{tok.text}:adj")

        # 将提取的形容词添加到集合中去重
        for a in adj:
            adj_set.add(a)

    return list(adj_set)


def get_nouns(nlp, doc):
    """
    Extract nouns from a list of tokens
    """
    nouns = []
    noun_set = set()
    for tok in doc:
        if tok.dep_ == "compound":
            comp_str = doc[tok.i : tok.head.i + 1]
            comp_str = to_singular(nlp, comp_str.text)
            for n in comp_str.split(" "):
                noun_set.add(f"{n}:noun")
            nouns.append(f"{comp_str}:noun")
    for tok in doc:
        if tok.pos_ == "NOUN":
            text = tok.text
            if tok.tag_ in {"NNS", "NNPS"}:
                text = tok.lemma_
            if text not in noun_set:
                nouns.append(f"{text}:noun")
    return nouns


def extract_concepts(nlp, texts):
    """
    Extract concepts (nouns and adjectives) from a list of texts
    """
    docs = nlp.pipe(texts)
    concepts = []
    for doc in docs:
        adjs = get_adj_pairs(doc)
        nouns = get_nouns(nlp, doc)
        for a in adjs:
            concepts.append(a)
        for n in nouns:
            concepts.append(n)
    return concepts


SPAWRIOUS_STOPWORDS = {
    "top",
    "middle",
    "side",
    "body",
    "bunch",
    "pile",
    "scene",
    "view",
    "image",
    "picture",
    "photo",
    "background",
}

SPAWRIOUS_CANONICAL_MAP = {
    "snow": "snow:noun",
    "snowy": "snow:noun",
    "mountain": "mountain:noun",
    "mountains": "mountain:noun",
    "beach": "beach:noun",
    "sand": "beach:noun",
    "sandy": "beach:noun",
    "jungle": "jungle:noun",
    "forest": "jungle:noun",
    "tree": "jungle:noun",
    "trees": "jungle:noun",
    "wood": "jungle:noun",
    "woods": "jungle:noun",
}

SPAWRIOUS_DEFAULT_IGNORE = {
    "dog",
    "frisbee",
    "front",
    "camera",
}

SPAWRIOUS_OPTIONAL_FILTER = {
    "black",
    "white",
    "brown",
    "small",
}

VALID_OPTIONAL_ACTIONS = {"keep", "ignore", "downweight"}


def is_spawrious_dataset(dataset):
    return dataset.startswith("spawrious_o2o_")


def is_metashift_dataset(dataset):
    return dataset.startswith("metashift")


def normalize_optional_action(action):
    action = (action or "keep").strip().lower()
    if action not in VALID_OPTIONAL_ACTIONS:
        raise ValueError(
            f"Unsupported optional action {action}. "
            f"Expected one of: {', '.join(sorted(VALID_OPTIONAL_ACTIONS))}"
        )
    return action


def build_spawrious_ignore_set(ignore_concepts=None):
    ignore_set = set(SPAWRIOUS_DEFAULT_IGNORE)
    if ignore_concepts:
        extra = {
            item.strip().lower()
            for item in ignore_concepts.split(",")
            if item.strip()
        }
        ignore_set.update(extra)
    return ignore_set


def should_ignore_base_concept(concept, ignore_set):
    text, _ = parse_concept(concept)
    return text in ignore_set


def should_ignore_optional_concept(concept, optional_action):
    if optional_action != "ignore":
        return False
    text, _ = parse_concept(concept)
    return text in SPAWRIOUS_OPTIONAL_FILTER


def concept_optional_weight(
    concept,
    optional_action="keep",
    optional_downweight=0.3,
):
    text, _ = parse_concept(concept)
    if text in SPAWRIOUS_OPTIONAL_FILTER and optional_action == "downweight":
        return float(optional_downweight)
    return 1.0


def parse_concept(concept):
    if ":" in concept:
        text, pos = concept.rsplit(":", 1)
        return text.strip().lower(), pos.strip().lower()
    return concept.strip().lower(), ""


def clean_spawrious_concept(concept):
    text, pos = parse_concept(concept)
    if not text:
        return None
    if text in SPAWRIOUS_STOPWORDS:
        return None
    canonical = SPAWRIOUS_CANONICAL_MAP.get(text)
    if canonical is not None:
        return canonical
    if pos:
        return f"{text}:{pos}"
    return text


def normalize_ignore_set(dataset, use_ignore_list=False, ignore_concepts=None):
    if is_spawrious_dataset(dataset):
        # Spawrious always uses the base ignore set.
        return build_spawrious_ignore_set(ignore_concepts)
    if not use_ignore_list:
        return set()
    if ignore_concepts:
        return {
            item.strip().lower()
            for item in ignore_concepts.split(",")
            if item.strip()
        }
    return set()


def maybe_clean_concepts(
    concept_arr,
    dataset,
    use_ignore_list=False,
    ignore_concepts=None,
    optional_action="keep",
    optional_downweight=0.3,
):
    optional_action = normalize_optional_action(optional_action)
    optional_downweight = float(optional_downweight)
    if not (0.0 < optional_downweight <= 1.0):
        raise ValueError("optional_downweight should be in (0, 1].")

    raw_counts = Counter()
    pre_ignore_counts = Counter()
    cleaned_counts = Counter()
    effective_counts = Counter()
    cleaned_arr = []
    changed_examples = []
    ignore_examples = []
    optional_examples = []
    ignore_set = normalize_ignore_set(
        dataset,
        use_ignore_list=use_ignore_list,
        ignore_concepts=ignore_concepts,
    )

    for idx, file_name, concepts in concept_arr:
        raw_counts.update(concepts)

        cleaned = []
        seen = set()
        for concept in concepts:
            if is_spawrious_dataset(dataset):
                cleaned_concept = clean_spawrious_concept(concept)
                if cleaned_concept is None:
                    if len(changed_examples) < 40:
                        changed_examples.append((concept, "__removed__"))
                    continue
                if cleaned_concept != concept and len(changed_examples) < 40:
                    changed_examples.append((concept, cleaned_concept))
            else:
                cleaned_concept = concept
            if cleaned_concept not in seen:
                cleaned.append(cleaned_concept)
                seen.add(cleaned_concept)

        pre_ignore_counts.update(cleaned)

        filtered = []
        for c in cleaned:
            if should_ignore_base_concept(c, ignore_set):
                if len(ignore_examples) < 40:
                    ignore_examples.append((c, "__ignored_base__"))
                continue
            if is_spawrious_dataset(dataset) and should_ignore_optional_concept(
                c, optional_action
            ):
                if len(optional_examples) < 40:
                    optional_examples.append((c, "__ignored_optional__"))
                continue
            if (
                is_spawrious_dataset(dataset)
                and optional_action == "downweight"
                and parse_concept(c)[0] in SPAWRIOUS_OPTIONAL_FILTER
                and len(optional_examples) < 40
            ):
                optional_examples.append(
                    (c, f"__downweighted_x{optional_downweight:.2f}__")
                )
            filtered.append(c)

        cleaned_arr.append((idx, normalize_image_key(file_name), filtered))
        cleaned_counts.update(filtered)
        for c in filtered:
            weight = concept_optional_weight(
                c,
                optional_action=optional_action
                if is_spawrious_dataset(dataset)
                else "keep",
                optional_downweight=optional_downweight,
            )
            effective_counts[c] += weight
    return (
        cleaned_arr,
        raw_counts,
        pre_ignore_counts,
        cleaned_counts,
        effective_counts,
        changed_examples,
        ignore_examples,
        optional_examples,
        ignore_set,
        optional_action,
        optional_downweight,
    )


def print_top_concepts(title, counts, top_k=12):
    print(title)
    for concept, count in counts.most_common(top_k):
        print(f"  {concept:<24} {count}")
    print()


def save_top_concepts_report(
    save_dir,
    caption_model,
    raw_counts,
    pre_ignore_counts,
    cleaned_counts,
    effective_counts,
    examples,
    ignore_examples,
    optional_examples,
    ignore_set,
    optional_action,
    optional_downweight,
    top_k=20,
):
    raw_top_path = os.path.join(save_dir, f"{caption_model}_top_concepts_raw.csv")
    pre_ignore_top_path = os.path.join(
        save_dir, f"{caption_model}_top_concepts_before_ignore.csv"
    )
    cleaned_top_path = os.path.join(
        save_dir, f"{caption_model}_top_concepts_after_ignore.csv"
    )
    legacy_cleaned_path = os.path.join(save_dir, f"{caption_model}_top_concepts_cleaned.csv")
    compare_path = os.path.join(save_dir, f"{caption_model}_top_concepts_ignore_compare.csv")
    effective_top_path = os.path.join(
        save_dir, f"{caption_model}_top_concepts_effective_weighted.csv"
    )
    preview_path = os.path.join(save_dir, f"{caption_model}_cleaning_preview.csv")

    with open(raw_top_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["concept", "count"])
        for concept, count in raw_counts.most_common(top_k):
            writer.writerow([concept, count])

    with open(pre_ignore_top_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["concept", "count"])
        for concept, count in pre_ignore_counts.most_common(top_k):
            writer.writerow([concept, count])

    with open(cleaned_top_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["concept", "count"])
        for concept, count in cleaned_counts.most_common(top_k):
            writer.writerow([concept, count])

    with open(legacy_cleaned_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["concept", "count"])
        for concept, count in cleaned_counts.most_common(top_k):
            writer.writerow([concept, count])

    with open(compare_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "before_ignore_concept",
                "before_ignore_count",
                "after_ignore_concept",
                "after_ignore_count",
            ]
        )
        before_top = pre_ignore_counts.most_common(top_k)
        after_top = cleaned_counts.most_common(top_k)
        for rank in range(top_k):
            before_c, before_n = ("", 0)
            after_c, after_n = ("", 0)
            if rank < len(before_top):
                before_c, before_n = before_top[rank]
            if rank < len(after_top):
                after_c, after_n = after_top[rank]
            writer.writerow([rank + 1, before_c, before_n, after_c, after_n])

    with open(effective_top_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["concept", "effective_count"])
        for concept, count in sorted(
            effective_counts.items(), key=lambda x: (-x[1], x[0])
        )[:top_k]:
            writer.writerow([concept, float(count)])

    with open(preview_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["before", "after"])
        for before, after in examples:
            writer.writerow([before, after])
        for before, after in ignore_examples:
            writer.writerow([before, after])
        for before, after in optional_examples:
            writer.writerow([before, after])

    ignore_info_path = os.path.join(save_dir, f"{caption_model}_ignore_list.txt")
    with open(ignore_info_path, "w", encoding="utf-8") as f:
        f.write("ignore_set:\n")
        for item in sorted(ignore_set):
            f.write(f"{item}\n")
        f.write("\n")
        f.write(f"optional_action: {optional_action}\n")
        f.write(f"optional_downweight: {optional_downweight}\n")
        f.write(
            "optional_filter_terms: "
            + ", ".join(sorted(SPAWRIOUS_OPTIONAL_FILTER))
            + "\n"
        )


def get_concept_embeddings(
    path,
    threshold=10,
    dataset=None,
    top_k_report=20,
    use_ignore_list=False,
    ignore_concepts=None,
    optional_action="keep",
    optional_downweight=0.3,
):
    """
    从指定路径的文件中提取概念并生成嵌入向量
    
    该函数读取包含提取概念的pickle文件，统计概念频率，构建词汇表，
    并为每个样本生成one-hot编码的概念嵌入向量。
    
    Args:
        path (str): 包含提取概念的pickle文件路径。文件名格式应为"{caption_model}_*.pickle"
        threshold (int): 概念频率阈值，只保留出现次数大于该阈值的概念。默认为10
    
    Returns:
        None: 函数将结果保存到文件中，不返回值
        
    保存的文件:
        - {caption_model}_vocab.pickle: 筛选后的词汇表（高频概念列表）
        - {caption_model}_img_embeddings.pickle: 概念嵌入矩阵，形状为(样本数, 词汇表大小)
    
    Note:
        - 输入文件格式：concept_arr是一个列表，每个元素是元组，其中第三个元素(concepts[2])包含概念列表
        - 输出嵌入矩阵采用one-hot编码，如果某个概念在样本中出现，对应位置设为1
    """
    # 从文件路径中提取caption模型名称
    caption_model = get_caption_tag(path)
    count = 0
    
    # 加载包含提取概念的pickle文件
    with open(path, "rb") as f:
        concept_arr = pickle.load(f)
    concept_arr = [
        (idx, normalize_image_key(file_name), concepts)
        for idx, file_name, concepts in concept_arr
    ]

    (
        concept_arr,
        raw_counts,
        pre_ignore_counts,
        cleaned_counts,
        effective_counts,
        changed_examples,
        ignore_examples,
        optional_examples,
        ignore_set,
        optional_action,
        optional_downweight,
    ) = maybe_clean_concepts(
        concept_arr,
        dataset,
        use_ignore_list=use_ignore_list,
        ignore_concepts=ignore_concepts,
        optional_action=optional_action,
        optional_downweight=optional_downweight,
    )
    if is_spawrious_dataset(dataset):
        print_top_concepts("Top concepts before cleaning:", raw_counts, top_k=top_k_report)
        print_top_concepts(
            "Top concepts before ignore-filter:",
            pre_ignore_counts,
            top_k=top_k_report,
        )
        print_top_concepts(
            "Top concepts after ignore-filter:",
            cleaned_counts,
            top_k=top_k_report,
        )
        print(f"Ignore list enabled: {bool(ignore_set)}")
        if ignore_set:
            print(f"Ignore terms: {', '.join(sorted(ignore_set))}")
        if changed_examples:
            print("Cleaning examples:")
            for before, after in changed_examples[:12]:
                print(f"  {before} -> {after}")
            print()
        if ignore_examples:
            print("Ignore-filter examples:")
            for before, after in ignore_examples[:12]:
                print(f"  {before} -> {after}")
            print()
        if optional_examples:
            print("Optional-filter examples:")
            for before, after in optional_examples[:12]:
                print(f"  {before} -> {after}")
            print()
    
    # 统计每个概念在所有样本中的出现频率
    concept_counts = {}
    for concepts in tqdm(concept_arr, desc="count concepts"):
        for c in concepts[2]:
            if c in concept_counts:
                concept_counts[c] += 1
            else:
                concept_counts[c] = 1
    
    # 将概念计数字典转换为列表，并按频率降序排序
    concept_counts = [(k, v) for k, v in concept_counts.items()]
    concept_counts = sorted(concept_counts, key=lambda x: -x[1])
    
    # 分离概念名称和对应的计数
    concepts = np.array([t[0] for t in concept_counts])
    counts = np.array([t[1] for t in concept_counts])
    effective = np.array(
        [effective_counts.get(c, float(n)) for c, n in concept_counts], dtype=float
    )
    
    # 根据阈值筛选高频概念，构建词汇表
    vocab = concepts[effective > threshold]

    vocab_size = len(vocab)
    print(
        f"vocab size is {vocab_size}|({len(concepts)}) ({vocab_size/len(concepts):.2f})"
    )
    
    # 保存词汇表到pickle文件
    save_dir = os.path.dirname(path)
    save_path = os.path.join(save_dir, f"{caption_model}_vocab.pickle")
    with open(save_path, "wb") as outfile:
        pickle.dump(vocab, outfile)

    embedding_keys = [normalize_image_key(row[1]) for row in concept_arr]
    duplicate_key_count = len(embedding_keys) - len(set(embedding_keys))
    if duplicate_key_count:
        msg = (
            f"Duplicate image keys detected while building concept embeddings: "
            f"{duplicate_key_count}"
        )
        if is_spawrious_dataset(dataset):
            raise ValueError(msg)
        print(f"WARNING: {msg}")
    
    # 创建概念到索引的映射字典
    concept2idx = {v: i for i, v in enumerate(vocab)}

    # 初始化概念嵌入矩阵（one-hot编码）
    concept_embeds = np.zeros((len(concept_arr), vocab_size))
    
    # 为每个样本生成one-hot编码的概念嵌入向量
    for idx, concepts in tqdm(enumerate(concept_arr), desc="Generate embeddings"):
        for c in concepts[2]:
            if c in concept2idx:
                concept_embeds[idx, concept2idx[c]] = 1
    
    # 保存概念嵌入矩阵到pickle文件
    save_dir = os.path.dirname(path)
    save_path = os.path.join(save_dir, f"{caption_model}_img_embeddings.pickle")
    with open(save_path, "wb") as outfile:
        pickle.dump(concept_embeds, outfile)
    key_save_path = os.path.join(save_dir, f"{caption_model}_img_embedding_keys.pickle")
    with open(key_save_path, "wb") as outfile:
        pickle.dump(embedding_keys, outfile)
    key_csv_path = os.path.join(save_dir, f"{caption_model}_img_embedding_keys.csv")
    with open(key_csv_path, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["row_index", "image_key"])
        for row_index, image_key in enumerate(embedding_keys):
            writer.writerow([row_index, image_key])
    if is_spawrious_dataset(dataset):
        save_top_concepts_report(
            save_dir,
            caption_model,
            raw_counts,
            pre_ignore_counts,
            cleaned_counts,
            effective_counts,
            changed_examples,
            ignore_examples,
            optional_examples,
            ignore_set,
            optional_action,
            optional_downweight,
            top_k=top_k_report,
        )


def get_concepts(caption_path, splits=0, split_idx=0):
    """
    Extract concepts from captions stored in a file specified by caption_path
    """
    save_dir = os.path.dirname(caption_path)
    caption_model = get_caption_tag(caption_path)
    save_path = os.path.join(
        save_dir, f"{caption_model}_extracted_concepts_{split_idx}_{splits}.pickle"
    )
    if os.path.exists(save_path):
        return save_path
    try:
        nlp = spacy.load("en_core_web_trf")
    except OSError:
        print("en_core_web_trf not found, falling back to en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    words_list = []
    with open(caption_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if len(row) < 3:
                continue
            file_name = normalize_image_key(row[0])
            caption = row[1].strip()
            words_list.append((i, file_name, caption))

    num = len(words_list)
    if splits > 0:
        num_per_split = num // splits
        start_idx = num_per_split * split_idx
        if split_idx == splits - 1:  # the last part
            end_idx = num
        else:
            end_idx = num_per_split * (split_idx + 1)
        print(
            f"[split_idx: {split_idx}] total: {num}, num_splits: {splits} num_per_split: {num_per_split}, range: {start_idx}-{end_idx}"
        )
    else:
        start_idx = 0
        end_idx = num
    sel_words_list = words_list[start_idx:end_idx]
    concepts_arr = []

    for eles in tqdm(sel_words_list):
        concepts = extract_concepts(nlp, [eles[2]])
        concepts = list(set(concepts))
        concepts_arr.append((eles[0], eles[1], concepts))

    with open(save_path, "wb") as outfile:
        pickle.dump(concepts_arr, outfile)
    return save_path


def get_data_folder(dataset):
    """
    Get the image folder and metadata file path for a dataset
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if dataset == "waterbirds":
        csv_path = os.path.join(
            base_dir, "waterbird_complete95_forest2water2", "metadata.csv"
        )
        img_path = os.path.join(base_dir, "waterbird_complete95_forest2water2", "images")
    elif dataset == "celeba":
        csv_path = os.path.join(base_dir, "celeba", "img_align_celeba", "metadata.csv")
        img_path = os.path.join(base_dir, "celeba", "img_align_celeba")
    elif dataset == "nico":
        csv_path = os.path.join(base_dir, "NICO", "multi_classification", "metadata.csv")
        img_path = os.path.join(base_dir, "NICO", "multi_classification")
    elif dataset == "imagenet-9":
        csv_path = os.path.join(base_dir, "imagenet", "metadata.csv")
        img_path = os.path.join(base_dir, "imagenet")
    elif is_spawrious_dataset(dataset):
        suffix = dataset.removeprefix("spawrious_o2o_")
        csv_path = os.path.join(
            base_dir, "SPUME-master", "data", f"{dataset}_metadata.csv"
        )
        img_path = os.path.join(base_dir, f"spawrious224__o2o_{suffix}")
    elif is_metashift_dataset(dataset):
        csv_path = os.path.join(base_dir, "datasets", "metashift", f"metadata_{dataset}.csv")
        img_path = os.path.join(base_dir, "datasets", "metashift")
    else:
        raise ValueError(f"Dataset {dataset} is not supported")
    return img_path, csv_path


def normalize_concept_model(model_name):
    normalized = model_name.strip().lower().replace("_", "-")
    if normalized == "vit-gpt2":
        return "vit-gpt2"
    if normalized == "blip":
        return "blip"
    raise ValueError(f"Captioning model {model_name} not supported")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="waterbirds", help="name of a dataset"
    )
    parser.add_argument(
        "--model", type=str, default="vit-gpt2", help="image2text model"
    )
    parser.add_argument(
        "--concept_model",
        type=str,
        default=None,
        help="concept model alias: vit_gpt2 / vit-gpt2 / blip",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="optional local directory for the captioning model",
    )
    parser.add_argument(
        "--caption_batch_size",
        type=int,
        default=8,
        help="batch size for caption generation",
    )
    parser.add_argument(
        "--caption_num_workers",
        type=int,
        default=0,
        help="num workers for caption generation dataloader",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="if > 0, only caption the first N non-test samples for debugging",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=1,
        help="print one progress log every N caption batches",
    )
    parser.add_argument(
        "--top_k_report",
        type=int,
        default=20,
        help="number of top concepts to print/save in cleaning reports",
    )
    parser.add_argument(
        "--use_ignore_list",
        action="store_true",
        help="enable custom ignore list for non-Spawrious datasets",
    )
    parser.add_argument(
        "--ignore_concepts",
        type=str,
        default="",
        help="comma-separated concept text to ignore; for Spawrious these are merged into default ignore terms",
    )
    parser.add_argument(
        "--spawrious_optional_action",
        type=str,
        default="downweight",
        choices=["keep", "ignore", "downweight"],
        help="optional treatment for Spawrious terms: black, white, brown, small",
    )
    parser.add_argument(
        "--spawrious_optional_downweight",
        type=float,
        default=0.3,
        help="effective weight for optional terms when action=downweight",
    )
    args = parser.parse_args()
    selected_model = args.concept_model if args.concept_model else args.model
    selected_model = normalize_concept_model(selected_model)

    if selected_model == "vit-gpt2":
        caption_model = VITGPT2_CAPTIONING(local_dir=args.model_dir)
    elif selected_model == "blip":
        caption_model = BLIP_CAPTIONING(local_dir=args.model_dir)
    else:
        raise ValueError(f"Captioning model {selected_model} not supported")

    data_folder, csv_path = get_data_folder(args.dataset)
    print(f"Process {args.dataset}")
    print(f"Concept model: {selected_model}")

    timer = utils.Timer()
    caption_path = caption_model.get_img_captions(
        data_folder,
        csv_path,
        batch_size=args.caption_batch_size,
        num_workers=args.caption_num_workers,
        max_samples=args.max_samples,
        log_every=args.log_every,
    )
    elapsed_time = timer.t()
    print(f"Time for captioning: {utils.time_str(elapsed_time)}")
    concept_path = get_concepts(caption_path)
    get_concept_embeddings(
        concept_path,
        threshold=10,
        dataset=args.dataset,
        top_k_report=args.top_k_report,
        use_ignore_list=args.use_ignore_list,
        ignore_concepts=args.ignore_concepts,
        optional_action=args.spawrious_optional_action,
        optional_downweight=args.spawrious_optional_downweight,
    )
    total_time = timer.t()
    print(f"Time for attribute extraction: {utils.time_str(total_time-elapsed_time)}")
    print(f"Total time: {utils.time_str(total_time)}")
