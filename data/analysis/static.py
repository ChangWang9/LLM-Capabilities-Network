# -*- coding: utf-8 -*-
"""
多能力归类 + 统计版
- 支持同一数据集对应多个能力：若数据集在 categories.json 的多个能力下出现，
  则它的文件与统计会同时归入这些能力。
- 特判 gap_coref -> GAP Coreference Dataset（前提：该数据集出现在 categories.json 中）
- 统计问/答长度（词数），并输出到 capability_dataset_stats.json
- 文件支持 .jsonl（逐行 JSON）与 .json（list 或 dict 包裹）

输出：
1) classified_by_capability.json          # 能力 -> 文件路径列表（去重）
2) dataset_key_to_capabilities.json       # 数据集短键 -> [能力, ...]
3) unmatched_files.json                   # 未匹配上的文件
4) capability_dataset_stats.json          # 能力 -> { 数据集短键 -> 统计 }
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict
from statistics import mean

QUESTION_KEYS = ["question", "prompt", "input", "instruction", "context", "text", "source", "query", "title"]
ANSWER_KEYS   = ["answer", "output", "target", "label", "response", "completion",
                 "answers", "labels", "targets", "ground_truth", "gold", "output_text", "y", "response_text", "choices"]

def normalize(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', s.lower())

def load_categories(categories_path: Path):
    with categories_path.open('r', encoding='utf-8-sig') as f:
        return json.load(f)

def build_dataset_index(categories_map: dict):
    """
    构建：dataset_norm -> set(capabilities)
    再加入常见别名（含 gap_coref 特判）映射到同一 set(capabilities)
    """
    ds_norm_to_caps = defaultdict(set)

    # 先把正式名打入映射
    for cap, datasets in categories_map.items():
        for ds in datasets:
            ds_norm = normalize(ds)
            ds_norm_to_caps[ds_norm].add(cap)

    # 常见别名（alias -> canonical dataset name）
    manual_aliases = {
        "coco": "COCO Captions",
        "cococaptions": "COCO Captions",
        "vg": "Visual Genome",
        "visualgenome": "Visual Genome",
        "f30k": "Flickr30k",
        "flickr30k": "Flickr30k",
        "tqa": "TriviaQA",
        "triviaqa": "TriviaQA",
        "mmlu": "MMLU",
        "math": "MATH",
        "alpacaeval": "AlpacaEval",
        "truthfulqa": "TruthfulQA",
        "lambada": "LAMBADA",
        "agent": "AGENT",
        # 关键：gap_coref 系列 → GAP Coreference Dataset
        "gap_coref": "GAP Coreference Dataset",
        "gapcoref": "GAP Coreference Dataset",
        "gapcoreferencedataset": "GAP Coreference Dataset",
    }

    # 将别名映射到 canonical 的能力集合
    for alias, canonical in manual_aliases.items():
        a = normalize(alias)
        c = normalize(canonical)
        if c in ds_norm_to_caps:
            # 只有 canonical 存在时才建立别名（避免把未知数据集“造”进来）
            ds_norm_to_caps[a] |= ds_norm_to_caps[c]

    return ds_norm_to_caps

def guess_dataset_norm_from_filename(filename_no_ext: str, ds_norm_to_caps: dict):
    """
    通过文件名推测 dataset_norm（归一化名）：
    1) 特判 gap_coref
    2) 全名归一化包含匹配（最长优先）
    3) token 精确匹配
    """
    fn_lower = filename_no_ext.lower()
    file_norm = normalize(filename_no_ext)

    # 特判 gap_coref
    if "gap_coref" in fn_lower or "gapcoref" in fn_lower:
        cand = normalize("GAP Coreference Dataset")
        if cand in ds_norm_to_caps:
            return cand
        # 若 categories.json 尚未包含该数据集，则返回 None（落入 unmatched）
        return None

    hits = [ds for ds in ds_norm_to_caps.keys() if ds and ds in file_norm]
    if hits:
        hits.sort(key=len, reverse=True)
        return hits[0]

    tokens = re.split(r'[_\-.]+', fn_lower)
    for t in tokens:
        t_norm = normalize(t)
        if t_norm in ds_norm_to_caps:
            return t_norm

    return None

def choose_dataset_key(filename_no_ext: str, ds_norm: str) -> str:
    """
    给统计/结果用的“短键”（更友好）：
    - 文件名包含 gap_coref/gapcoref -> 'gap_coref'
    - 否则优先挑选能代表数据集的 token（如 mmlu、glue、triviaqa）
    - 实在不行就用 ds_norm
    """
    fn = filename_no_ext.lower()
    if "gap_coref" in fn or "gapcoref" in fn:
        return "gap_coref"

    tokens = re.split(r'[_\-.]+', fn)
    for t in tokens:
        t_norm = normalize(t)
        if t_norm and ds_norm and (t_norm in ds_norm or ds_norm in t_norm):
            return t
    return ds_norm

def iter_samples_from_file(path: Path):
    """
    统一迭代样本项（字典）。支持：
    - JSONL: 每行一个 JSON
    - JSON: 顶层 list，或 dict 包含 data/train/examples/instances/items/records/samples
    """
    if path.suffix.lower() == ".jsonl":
        with path.open('r', encoding='utf-8-sig') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    yield obj if isinstance(obj, dict) else {"_": obj}
                except Exception:
                    continue
    else:  # .json
        with path.open('r', encoding='utf-8-sig') as f:
            try:
                obj = json.load(f)
            except Exception:
                return
        if isinstance(obj, list):
            for it in obj:
                yield it if isinstance(it, dict) else {"_": it}
        elif isinstance(obj, dict):
            for key in ["data", "train", "examples", "instances", "items", "records", "samples"]:
                if key in obj and isinstance(obj[key], list):
                    for it in obj[key]:
                        yield it if isinstance(it, dict) else {"_": it}
                    return
            yield obj  # 兜底

def first_string(value):
    """从 value 提取字符串；list 取第一个字符串；dict 不转为字符串（除 choices 特判）。"""
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        return s if s else None
    if isinstance(value, list):
        for v in value:
            s = first_string(v)
            if s:
                return s
        return None
    if isinstance(value, (int, float, bool)):
        return str(value)
    return None

def from_choices(value):
    """
    针对 'choices' 常见结构做提取：
    - list[str] -> 第一个非空
    - list[dict] -> 找 'text'/'label'/'answer'/'content'/'value'/'option' 等
    """
    if isinstance(value, list):
        # 先找纯字符串
        for v in value:
            if isinstance(v, str) and v.strip():
                return v.strip()
        # 再找 dict 常见字段
        for v in value:
            if isinstance(v, dict):
                for k in ["text", "label", "answer", "content", "value", "option", "message"]:
                    if k in v and isinstance(v[k], str) and v[k].strip():
                        return v[k].strip()
    return None

def extract_qa(sample: dict):
    """从样本中提取 (question_str, answer_str)。"""
    q = None
    a = None
    for k in QUESTION_KEYS:
        if k in sample:
            q = first_string(sample[k])
            if q:
                break
    for k in ANSWER_KEYS:
        if k in sample:
            if k == "choices":
                a = from_choices(sample[k]) or first_string(sample[k])
            else:
                a = first_string(sample[k])
            if a:
                break
    return q, a

def word_len(s: str) -> int:
    return len(s.split()) if s else 0

def merge_acc(acc_all, q, a):
    acc_all["count"] += 1
    if q:
        acc_all["q"].append(word_len(q))
    if a:
        acc_all["a"].append(word_len(a))

def finalize_stat(acc):
    def pack(arr):
        if not arr:
            return None, None, None
        return float(mean(arr)), int(min(arr)), int(max(arr))
    q_avg, q_min, q_max = pack(acc["q"])
    a_avg, a_min, a_max = pack(acc["a"])
    return {
        "count": acc["count"],
        "question_avg_len": q_avg,
        "question_min_len": q_min,
        "question_max_len": q_max,
        "answer_avg_len": a_avg,
        "answer_min_len": a_min,
        "answer_max_len": a_max,
    }

def main():
    parser = argparse.ArgumentParser(description="Classify datasets (multi-capabilities) and compute QA stats")
    parser.add_argument(
        "--categories",
        default=r"C:\Users\Merlin\Desktop\Grade3\GSAI\Capability_Graph_of_LLM\LLM-Capabilities-Network\data\analysis\categories.json",
        help="Path to categories.json",
    )
    parser.add_argument(
        "--train-dir",
        default=r"C:\Users\Merlin\Desktop\Grade3\GSAI\Capability_Graph_of_LLM\LLM-Capabilities-Network\data\train",
        help="Directory containing training dataset files",
    )
    parser.add_argument(
        "--out-dir",
        default=r"C:\Users\Merlin\Desktop\Grade3\GSAI\Capability_Graph_of_LLM\LLM-Capabilities-Network\data\analysis",
        help="Directory to write output JSONs",
    )
    parser.add_argument(
        "--patterns",
        nargs="*",
        default=["*.json", "*.jsonl"],
        help="Glob patterns to include (space-separated)",
    )
    args = parser.parse_args()

    categories_path = Path(args.categories)
    train_dir = Path(args.train_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    assert categories_path.exists(), f"categories.json 未找到: {categories_path}"
    assert train_dir.exists(), f"训练集目录未找到: {train_dir}"

    categories_map = load_categories(categories_path)
    ds_norm_to_caps = build_dataset_index(categories_map)

    # 汇总容器
    capability_to_files = defaultdict(set)       # 能力 -> set(文件)
    dataset_key_to_capabilities = defaultdict(set)  # 短键 -> set(能力)
    unmatched_files = []
    dataset_key_acc = defaultdict(lambda: {"q": [], "a": [], "count": 0})  # 数据集短键 -> 累积统计
    ds_key_to_norms = defaultdict(set)  # 短键 -> {ds_norm}（同一短键可能指向同一 canonical/alias）
    ds_norm_for_file = {}              # 文件 -> ds_norm（主要用于检索其能力集）

    # 遍历文件：先做“数据集级别”的统计累计 & 记录它对应的能力集合
    all_files = []
    for pat in args.patterns:
        all_files.extend(train_dir.rglob(pat))

    for f in sorted(set(all_files)):
        if not f.is_file():
            continue
        base_no_ext = f.name.rsplit('.', 1)[0]
        ds_norm = guess_dataset_norm_from_filename(base_no_ext, ds_norm_to_caps)

        if ds_norm is None:
            unmatched_files.append(str(f))
            continue

        caps = ds_norm_to_caps.get(ds_norm)
        if not caps:
            unmatched_files.append(str(f))
            continue

        # 选择短键并累计统计
        ds_key = choose_dataset_key(base_no_ext, ds_norm)
        ds_key_to_norms[ds_key].add(ds_norm)
        ds_norm_for_file[str(f)] = ds_norm

        # 统计该文件的 QA
        acc = dataset_key_acc[ds_key]
        for sample in iter_samples_from_file(f):
            if isinstance(sample, dict):
                q, a = extract_qa(sample)
                merge_acc(acc, q, a)

        # 文件归入所有能力（去重用 set）
        for cap in caps:
            capability_to_files[cap].add(str(f))
            dataset_key_to_capabilities[ds_key].add(cap)

    # 整理统计（先按数据集短键 finalize，再分发到每个能力）
    dataset_key_stats = {k: finalize_stat(v) for k, v in dataset_key_acc.items()}

    capability_dataset_stats = defaultdict(dict)
    for ds_key, caps in dataset_key_to_capabilities.items():
        stat = dataset_key_stats.get(ds_key, {
            "count": 0,
            "question_avg_len": None, "question_min_len": None, "question_max_len": None,
            "answer_avg_len": None, "answer_min_len": None, "answer_max_len": None,
        })
        for cap in caps:
            capability_dataset_stats[cap][ds_key] = stat

    # 写出文件（set 转 list）
    out_by_cap = out_dir / "classified_by_capability.json"
    out_ds2caps = out_dir / "dataset_key_to_capabilities.json"
    out_unmatched = out_dir / "unmatched_files.json"
    out_stats = out_dir / "capability_dataset_stats.json"

    with out_by_cap.open('w', encoding='utf-8') as f:
        json.dump({cap: sorted(list(paths)) for cap, paths in capability_to_files.items()},
                  f, ensure_ascii=False, indent=2)
    with out_ds2caps.open('w', encoding='utf-8') as f:
        json.dump({k: sorted(list(v)) for k, v in dataset_key_to_capabilities.items()},
                  f, ensure_ascii=False, indent=2)
    with out_unmatched.open('w', encoding='utf-8') as f:
        json.dump(unmatched_files, f, ensure_ascii=False, indent=2)
    with out_stats.open('w', encoding='utf-8') as f:
        json.dump(capability_dataset_stats, f, ensure_ascii=False, indent=2)

    print("处理完成。输出：")
    print(f"- {out_by_cap}")
    print(f"- {out_ds2caps}  <-- 数据集短键 -> 能力列表")
    print(f"- {out_unmatched}  (未匹配文件数：{len(unmatched_files)})")
    print(f"- {out_stats}  <-- 统计结果（按能力 -> 数据集短键）")
    if unmatched_files:
        print("提示：若未匹配包含新的缩写/别名，请在脚本内的 manual_aliases 中扩充；或补充到 categories.json。")

if __name__ == "__main__":
    main()
