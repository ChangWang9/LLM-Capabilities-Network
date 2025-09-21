# -*- coding: utf-8 -*-
"""
static_with_coverage.py

在原有 static.py 的基础上，新增生成最终覆盖率 JSON（含 dataset_count）：
- 读取 categories.json + 扫描 train/ 文件 -> 完成多能力归类与数据集映射
- 载入“新能力体系（8 大类）”，用同义词映射把“新表能力”对齐到 categories.json 的能力键
- 按你的 68 项“语言能力清单”做覆盖判断（Exact / Synonym / Split），并统计：
  * summary（语言/多模态/总体）
  * language_capabilities 各大类明细（每个能力：status + dataset_count）

输出文件（位于 --out-dir）：
1) classified_by_capability.json           # 能力 -> 文件路径列表（去重）
2) dataset_key_to_capabilities.json        # 数据集短键 -> [能力, ...]
3) unmatched_files.json                    # 未匹配上的文件
4) capability_dataset_stats.json           # 能力 -> { 数据集短键 -> 统计 }
5) capability_coverage_summary.json        # ✅ 你要的最终覆盖率结果（含 dataset_count）
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict
from statistics import mean

# -----------------------------
# 可按需扩展/修改的配置
# -----------------------------

# 你的“68 项语言能力清单”
USER_LANG_CAPS = [
    "Abstract reasoning capability","Alignment capability","Analytical thinking capability","Attention capability",
    "Bias mitigation capability","Causal learning capability","Causal reasoning capability","Chemical understanding capability",
    "Classification capability","Code generation capability","Common sense reasoning capability","Compositional capability",
    "Comprehension and interpretation capability","Context understanding capability","Contextual learning capability",
    "Contextual reasoning capability","Controllable generation capability","Counterfactual reasoning capability",
    "Creative generation capability","Critique capability","Data filtering capability","Domain reasoning capability",
    "Ethical reasoning capability","Evaluation capability","Execution reasoning capability","Fairness capability",
    "Few-shot learning capability","General generative capability","Generalization capability","High-quality text generation capability",
    "Induction and inference capability","Information retrieval capability","Instruction following capability",
    "Intention detection capability","Judgment capability","Knowledge extraction capability","Knowledge integration capability",
    "Knowledge learning capability","Knowledge recall capability","Language understanding capability","Linguistic capability",
    "Logical reasoning capability","Long context capability","Mathematical computation capability","Mathematical reasoning capability",
    "Memorization capability","Multilingual capability","Optimization capability","Privacy capability","Problem solving capability",
    "Programming capability","Prompt learning capability","Question raising capability","Reflection capability","Representation capability",
    "Representation learning capability","Robustness capability","SQL capability","Safety capability","Self-correction capability",
    "Sequence generation capability","Structure generation capability","Temporal reasoning capability","Text generation capability",
    "Transfer capability","Trustworthiness capability","Verification capability","Zero-shot learning capability"
]

# 新能力体系（8 大类）
TAXONOMY = {
    "1. Reasoning Capabilities": [
        "Causal reasoning capability","Mathematical reasoning capability","Temporal reasoning capability",
        "Logical reasoning capability","Contextual reasoning capability","Counterfactual reasoning capability",
        "Induction and inference capability","Problem solving capability","Common sense reasoning capability",
        "Analytical thinking capability","Abstract reasoning capability","Domain reasoning capability","Execution reasoning capability"
    ],
    "2. Generation Capabilities": [
        "Text generation capability","Image generation capability","Code generation capability","Creative generation capability",
        "Structure generation capability","High-quality text generation capability","Sequence generation capability",
        "Story generation capability","Controllable generation capability","General generative capability"
    ],
    "3. Understanding Capabilities": [
        "Language understanding capability","Visual understanding capability","Spatial understanding capability",
        "String processing capability","Speech comprehension capability","Recognition capability","Perception capability",
        "Comprehension and interpretation capability","Grounding capability","Multimodal understanding capability",
        "Context understanding capability","Video understanding capability","Chemical understanding capability",
        "Physical world understanding capability","Prompt understanding capability"
    ],
    "4. Learning Capabilities": [
        "Generalization capability","Transfer capability","Contextual learning capability","Adaptation capability",
        "Memorization capability","Few-shot learning capability","Zero-shot learning capability","Continuous learning capability",
        "Knowledge learning capability","Representation learning capability","Prompt learning capability","Causal learning capability"
    ],
    "5. Alignment & Safety Capabilities": [
        "Safety capability","Instruction following capability","Alignment capability","Robustness capability",
        "Bias mitigation capability","Ethical reasoning capability","Trustworthiness capability","Fairness capability","Privacy capability"
    ],
    "6. Domain-Specific Capabilities": [
        "Translation capability","Multimodal capability","Programming capability","Mathematical computation capability",
        "Visual capability","Linguistic capability","Audio-visual capability","Spatial reasoning capability","Graph understanding capability",
        "SQL capability","Image description capability","Multilingual capability","Image matching capability","Label recognition capability",
        "Data filtering capability"
    ],
    "7. Meta-Cognitive Capabilities": [
        "Planning capability","Evaluation capability","Judgment capability","Reflection capability","Optimization capability",
        "Intention detection capability","Self-correction capability","Tool usage capability","Monitoring capability",
        "Function calling capability","Verification capability","Question raising capability","Critique capability"
    ],
    "8. Other Notable Capabilities": [
        "Language modeling capability","Knowledge recall capability","Knowledge integration capability","Knowledge extraction capability",
        "Information retrieval capability","Attention capability","Expression capability","Representation capability","Long context capability",
        "Personalization capability","Communication capability","Compositional capability","Streaming capability","Classification capability",
        "Localization capability"
    ],
}

# 中文分组名
CN_SECTION_NAMES = {
    "1. Reasoning Capabilities": "推理类能力",
    "2. Generation Capabilities": "生成类能力",
    "3. Understanding Capabilities": "理解类能力",
    "4. Learning Capabilities": "学习类能力",
    "5. Alignment & Safety Capabilities": "对齐与安全能力",
    "6. Domain-Specific Capabilities": "领域类能力",
    "7. Meta-Cognitive Capabilities": "元认知能力",
    "8. Other Notable Capabilities": "其他能力",
}

# “新表能力名” -> categories.json 里的能力名集合（同义/拆分映射）；默认映射到自身
# 你可在此继续补充映射以提升匹配率
CAPABILITY_SYNONYMS_TO_CATEGORIES = {
    "Analytical thinking capability": ["Analytical reasoning capability"],
    "High-quality text generation capability": ["Quality text generation capability","High-quality text generation capability"],
    "General generative capability": ["Generative capability","General generative capability"],
    "Induction and inference capability": ["Induction capability","Inference capability"],
    "Mathematical computation capability": ["Numerical computation capability","Mathematical computation capability"],
    "Programming capability": ["Coding capability","Programming capability"],
    "Comprehension and interpretation capability": ["Comprehension capability","Interpretation capability","Comprehension and interpretation capability"],
    "Robustness capability": ["Robust capability","Robustness capability"],
    "Contextual learning capability": ["Context learning capability","In-context learning capability","Contextual learning capability"],
    "Continuous learning capability": ["Continual learning capability","Lifelong learning capability","Continuous learning capability"],
    "Image description capability": ["Image caption capability","Image captioning capability","Image description capability"],
    "Tool usage capability": ["Tool use capability","Tool utilization capability","Tool usage capability"],
    "Language understanding capability": ["Natural language understanding capability","Text understanding capability","Language understanding capability"],
    # 你也可以把“Domain reasoning capability”等未在 categories.json 出现的能力，映射到接近的能力上
    # "Domain reasoning capability": ["Domain knowledge learning capability","Domain adaptation capability"],
}

# -----------------------------
# 以下是静态.py 的主体 + 产出覆盖率 JSON
# -----------------------------

QUESTION_KEYS = ["question","prompt","input","instruction","context","text","source","query","title"]
ANSWER_KEYS   = ["answer","output","target","label","response","completion","answers","labels","targets","ground_truth","gold","output_text","y","response_text","choices"]

def normalize(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', s.lower())

def is_multimodal(name: str) -> bool:
    mm_keywords = ["image","video","visual","audio","speech","multimodal","caption","matching","localization","spatial",
                   "perception","recognition","modality","grounding","physical world","ocr","camera","vision"]
    return any(k in name.lower() for k in mm_keywords)

def load_categories(categories_path: Path):
    with categories_path.open('r', encoding='utf-8-sig') as f:
        return json.load(f)

def build_dataset_index(categories_map: dict):
    ds_norm_to_caps = defaultdict(set)
    for cap, datasets in categories_map.items():
        for ds in datasets:
            ds_norm = normalize(ds)
            ds_norm_to_caps[ds_norm].add(cap)

    manual_aliases = {
        "coco": "COCO Captions","cococaptions":"COCO Captions",
        "vg":"Visual Genome","visualgenome":"Visual Genome",
        "f30k":"Flickr30k","flickr30k":"Flickr30k",
        "tqa":"TriviaQA","triviaqa":"TriviaQA",
        "mmlu":"MMLU","math":"MATH",
        "alpacaeval":"AlpacaEval","truthfulqa":"TruthfulQA",
        "lambada":"LAMBADA","agent":"AGENT",
        "gap_coref":"GAP Coreference Dataset","gapcoref":"GAP Coreference Dataset","gapcoreferencedataset":"GAP Coreference Dataset",
    }
    for alias, canonical in manual_aliases.items():
        a = normalize(alias); c = normalize(canonical)
        if c in ds_norm_to_caps:
            ds_norm_to_caps[a] |= ds_norm_to_caps[c]
    return ds_norm_to_caps

def guess_dataset_norm_from_filename(filename_no_ext: str, ds_norm_to_caps: dict):
    fn_lower = filename_no_ext.lower()
    file_norm = normalize(filename_no_ext)
    if "gap_coref" in fn_lower or "gapcoref" in fn_lower:
        cand = normalize("GAP Coreference Dataset")
        if cand in ds_norm_to_caps:
            return cand
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
    if path.suffix.lower() == ".jsonl":
        with path.open('r', encoding='utf-8-sig') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                    yield obj if isinstance(obj, dict) else {"_": obj}
                except Exception:
                    continue
    else:
        with path.open('r', encoding='utf-8-sig') as f:
            try:
                obj = json.load(f)
            except Exception:
                return
        if isinstance(obj, list):
            for it in obj:
                yield it if isinstance(it, dict) else {"_": it}
        elif isinstance(obj, dict):
            for key in ["data","train","examples","instances","items","records","samples"]:
                if key in obj and isinstance(obj[key], list):
                    for it in obj[key]:
                        yield it if isinstance(it, dict) else {"_": it}
                    return
            yield obj

def first_string(value):
    if value is None: return None
    if isinstance(value, str):
        s = value.strip(); return s if s else None
    if isinstance(value, list):
        for v in value:
            s = first_string(v)
            if s: return s
        return None
    if isinstance(value, (int,float,bool)): return str(value)
    return None

def from_choices(value):
    if isinstance(value, list):
        for v in value:
            if isinstance(v, str) and v.strip(): return v.strip()
        for v in value:
            if isinstance(v, dict):
                for k in ["text","label","answer","content","value","option","message"]:
                    if k in v and isinstance(v[k], str) and v[k].strip():
                        return v[k].strip()
    return None

def extract_qa(sample: dict):
    q = None; a = None
    for k in QUESTION_KEYS:
        if k in sample:
            q = first_string(sample[k])
            if q: break
    for k in ANSWER_KEYS:
        if k in sample:
            a = from_choices(sample[k]) if k=="choices" else first_string(sample[k])
            if a: break
    return q, a

def word_len(s: str) -> int:
    return len(s.split()) if s else 0

def merge_acc(acc_all, q, a):
    acc_all["count"] += 1
    if q: acc_all["q"].append(word_len(q))
    if a: acc_all["a"].append(word_len(a))

def finalize_stat(acc):
    from statistics import mean
    def pack(arr):
        if not arr: return None, None, None
        return float(mean(arr)), int(min(arr)), int(max(arr))
    q_avg, q_min, q_max = pack(acc["q"])
    a_avg, a_min, a_max = pack(acc["a"])
    return {"count": acc["count"],"question_avg_len": q_avg,"question_min_len": q_min,"question_max_len": q_max,
            "answer_avg_len": a_avg,"answer_min_len": a_min,"answer_max_len": a_max}

def build_newtable_and_coverage(out_dir: Path, dataset_key_to_capabilities: dict):
    """
    依据 TAXONOMY + USER_LANG_CAPS + 同义映射，生成最终 coverage JSON（含 dataset_count）
    dataset_count 计算：将新表能力映射到 categories 能力集合，统计该集合所覆盖到的“数据集短键”数量（来自 dataset_key_to_capabilities）
    """
    def norm(s): return s.lower().strip()

    # 语言 / 多模态总表
    flat_caps = []
    for section, items in TAXONOMY.items():
        for c in items:
            flat_caps.append((section, c, "multimodal" if is_multimodal(c) else "language"))

    language_caps = [(sec,c) for (sec,c,t) in flat_caps if t=="language"]
    multimodal_caps = [(sec,c) for (sec,c,t) in flat_caps if t=="multimodal"]

    # 用户 68 项（含同义目标拆分的扩展）
    user_set = {norm(x) for x in USER_LANG_CAPS}
    for src, targets in CAPABILITY_SYNONYMS_TO_CATEGORIES.items():
        if src in USER_LANG_CAPS:
            for t in targets:
                user_set.add(norm(t))

    # “新表能力名” -> “categories 能力名集合”
    def map_to_category_caps(cap_name: str):
        mapped = CAPABILITY_SYNONYMS_TO_CATEGORIES.get(cap_name, None)
        if mapped:
            return set(mapped)
        return {cap_name}

    # 计算 dataset_count
    # dataset_key_to_capabilities: ds_key -> [capA, capB, ...]（来自上游分类）
    # 对于一个新表能力 cap_new：把它映射到一个 categories 能力集合 targets，
    # 如果某 ds_key 的能力集合与 targets 有交集，则该 ds_key 计入 cap_new 的 dataset_count（去重）
    cap_to_dskeys = defaultdict(set)
    for ds_key, caps in dataset_key_to_capabilities.items():
        caps_set = set(caps)
        for section, items in TAXONOMY.items():
            for cap_new in items:
                targets = map_to_category_caps(cap_new)
                if caps_set & targets:
                    cap_to_dskeys[cap_new].add(ds_key)

    # 覆盖判定：用户 68 清单对“新表能力”是否覆盖（同义算覆盖）
    def covered(cap_name: str) -> bool:
        if norm(cap_name) in user_set:
            return True
        # 如果 cap_name 作为同义映射源，且用户列出了其同义目标
        mapped = CAPABILITY_SYNONYMS_TO_CATEGORIES.get(cap_name, [])
        for t in mapped:
            if norm(t) in user_set:
                return True
        return False

    # 汇总 summary
    lang_total = len(language_caps)
    lang_covered = sum(1 for _, c in language_caps if covered(c))
    lang_uncovered = lang_total - lang_covered
    lang_rate = f"{(lang_covered / lang_total * 100):.1f}%" if lang_total else "0%"

    mm_total = len(multimodal_caps)
    mm_covered = 0  # 用户列表是语言能力
    mm_uncovered = mm_total - mm_covered
    mm_rate = f"{(mm_covered / mm_total * 100):.1f}%" if mm_total else "0%"

    overall_total = lang_total + mm_total
    overall_covered = lang_covered + mm_covered
    overall_uncovered = overall_total - overall_covered
    overall_rate = f"{(overall_covered / overall_total * 100):.1f}%" if overall_total else "0%"

    # 逐大类（只统计“语言类能力”）
    def section_obj(section_name: str):
        items = [c for (sec,c,t) in flat_caps if sec==section_name and t=="language"]
        total = len(items)
        covered_cnt = sum(1 for c in items if covered(c))
        rate = f"{(covered_cnt / total * 100):.1f}%" if total else "0%"
        caps_list = []
        for c in items:
            caps_list.append({
                "name": c,
                "status": "covered" if covered(c) else "uncovered",
                "dataset_count": len(cap_to_dskeys.get(c, set()))
            })
        return {
            "category_name": CN_SECTION_NAMES.get(section_name, section_name),
            "total": total,
            "covered": covered_cnt,
            "coverage_rate": rate,
            "capabilities": caps_list
        }

    language_detail = {
        "reasoning": section_obj("1. Reasoning Capabilities"),
        "generation": section_obj("2. Generation Capabilities"),
        "understanding": section_obj("3. Understanding Capabilities"),
        "learning": section_obj("4. Learning Capabilities"),
        "alignment_and_safety": section_obj("5. Alignment & Safety Capabilities"),
        "domain_specific": section_obj("6. Domain-Specific Capabilities"),
        "meta_cognitive": section_obj("7. Meta-Cognitive Capabilities"),
        "other": section_obj("8. Other Notable Capabilities"),
    }

    result = {
        "summary": {
            "language_capabilities": {
                "total": lang_total,
                "covered": lang_covered,
                "uncovered": lang_uncovered,
                "coverage_rate": lang_rate
            },
            "multimodal_capabilities": {
                "total": mm_total,
                "covered": mm_covered,
                "uncovered": mm_uncovered,
                "coverage_rate": mm_rate
            },
            "overall": {
                "total_capabilities": overall_total,
                "covered_capabilities": overall_covered,
                "uncovered_capabilities": overall_uncovered,
                "coverage_rate": overall_rate
            }
        },
        "language_capabilities": language_detail
    }

    out_path = out_dir / "capability_coverage_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"- {out_path}  <-- 覆盖率结果（含 dataset_count）已生成")

def main():
    parser = argparse.ArgumentParser(description="Classify datasets (multi-capabilities), compute stats and build coverage JSON")
    parser.add_argument("--categories", default=r"C:\Users\Merlin\Desktop\Grade3\GSAI\Capability_Graph_of_LLM\LLM-Capabilities-Network\data\analysis\categories.json", help="Path to categories.json")
    parser.add_argument("--train-dir", default=r"C:\Users\Merlin\Desktop\Grade3\GSAI\Capability_Graph_of_LLM\LLM-Capabilities-Network\data\train", help="Directory containing training dataset files")
    parser.add_argument("--out-dir", default=r"C:\Users\Merlin\Desktop\Grade3\GSAI\Capability_Graph_of_LLM\LLM-Capabilities-Network\data\analysis", help="Directory to write output JSONs")
    parser.add_argument("--patterns", nargs="*",
                        default=["*.json", "*.jsonl"],
                        help="Glob patterns to include (space-separated)")
    args = parser.parse_args()

    categories_path = Path(args.categories)
    train_dir = Path(args.train_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    assert categories_path.exists(), f"categories.json 未找到: {categories_path}"
    assert train_dir.exists(), f"训练集目录未找到: {train_dir}"

    # 1) 读取 categories，构建数据集归一化索引
    categories_map = load_categories(categories_path)
    ds_norm_to_caps = build_dataset_index(categories_map)

    # 2) 扫描 train/，完成“数据集短键 -> 能力集合”的映射，同时保留能力 -> 文件列表
    capability_to_files = defaultdict(set)
    dataset_key_to_capabilities = defaultdict(set)
    unmatched_files = []
    dataset_key_acc = defaultdict(lambda: {"q": [], "a": [], "count": 0})

    all_files = []
    for pat in args.patterns:
        all_files.extend(train_dir.rglob(pat))

    for f in sorted(set(all_files)):
        if not f.is_file(): continue
        base_no_ext = f.name.rsplit('.', 1)[0]
        ds_norm = guess_dataset_norm_from_filename(base_no_ext, ds_norm_to_caps)
        if ds_norm is None:
            unmatched_files.append(str(f)); continue

        caps = ds_norm_to_caps.get(ds_norm)
        if not caps:
            unmatched_files.append(str(f)); continue

        ds_key = choose_dataset_key(base_no_ext, ds_norm)

        # 统计该文件的 QA（可选，用不到 dataset_count，但保留与原功能一致）
        acc = dataset_key_acc[ds_key]
        for sample in iter_samples_from_file(f):
            if isinstance(sample, dict):
                q, a = extract_qa(sample)
                merge_acc(acc, q, a)

        for cap in caps:
            capability_to_files[cap].add(str(f))
            dataset_key_to_capabilities[ds_key].add(cap)

    dataset_key_stats = {k: finalize_stat(v) for k, v in dataset_key_acc.items()}

    capability_dataset_stats = defaultdict(dict)
    for ds_key, caps in dataset_key_to_capabilities.items():
        stat = dataset_key_stats.get(ds_key, {"count": 0,"question_avg_len": None,"question_min_len": None,"question_max_len": None,
                                              "answer_avg_len": None,"answer_min_len": None,"answer_max_len": None})
        for cap in caps:
            capability_dataset_stats[cap][ds_key] = stat

    # 3) 写出中间文件（便于检查）
    out_by_cap = out_dir / "classified_by_capability.json"
    out_ds2caps = out_dir / "dataset_key_to_capabilities.json"
    out_unmatched = out_dir / "unmatched_files.json"
    out_stats = out_dir / "capability_dataset_stats.json"

    with out_by_cap.open('w', encoding='utf-8') as f:
        json.dump({cap: sorted(list(paths)) for cap, paths in capability_to_files.items()}, f, ensure_ascii=False, indent=2)
    with out_ds2caps.open('w', encoding='utf-8') as f:
        json.dump({k: sorted(list(v)) for k, v in dataset_key_to_capabilities.items()}, f, ensure_ascii=False, indent=2)
    with out_unmatched.open('w', encoding='utf-8') as f:
        json.dump(unmatched_files, f, ensure_ascii=False, indent=2)
    with out_stats.open('w', encoding='utf-8') as f:
        json.dump(capability_dataset_stats, f, ensure_ascii=False, indent=2)

    print("中间结果：")
    print(f"- {out_by_cap}")
    print(f"- {out_ds2caps}  <-- 数据集短键 -> 能力列表")
    print(f"- {out_unmatched}  (未匹配文件数：{len(unmatched_files)})")
    print(f"- {out_stats}  <-- 能力 -> 数据集短键 -> 统计")

    # 4) 生成最终覆盖率 JSON（含 dataset_count）
    build_newtable_and_coverage(out_dir, json.loads(out_ds2caps.read_text(encoding='utf-8')))

if __name__ == "__main__":
    main()
