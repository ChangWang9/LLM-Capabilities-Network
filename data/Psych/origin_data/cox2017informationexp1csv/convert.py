import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

# --------- 基础正则 ----------
STUDY_HEAD = re.compile(r'You study the following 20 word pairs:\s*', re.M)
PAIR_LINE = re.compile(r'^\s*([A-Z][A-Z\-]+),\s*([A-Z][A-Z\-]+)\s*$', re.M)

# 从原文尾部那段元数据里兜底提取（如果对象里没有）
META_PATTS = {
    'participant': re.compile(r'"participant"\s*:\s*"([^"]+)"'),
    'experiment':  re.compile(r'"experiment"\s*:\s*"([^"]+)"'),
    'split':       re.compile(r'"split"\s*:\s*"([^"]+)"'),
}

def load_json_any(path: str) -> List[Dict[str, Any]]:
    """支持 JSON 数组 / 顶层带键JSON / JSONL"""
    p = Path(path)
    text = p.read_text(encoding='utf-8')
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # 尝试找一个列表型字段
            for v in data.values():
                if isinstance(v, list):
                    return v
            # 不是数组：尝试把整个对象作为单条
            return [data]
    except json.JSONDecodeError:
        # JSONL
        out = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return out
    return []

def pick_longest_string_field(obj: Dict[str, Any]) -> str:
    """当没有明确 text 字段时，挑最长的字符串字段当原文。"""
    best = ''
    for k, v in obj.items():
        if isinstance(v, str) and len(v) > len(best):
            best = v
    return best

def extract_meta_fallback(raw_text: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """如果 meta 缺失，从原文中兜底抽取。"""
    m = dict(meta)
    for key, patt in META_PATTS.items():
        if not m.get(key):
            mm = patt.search(raw_text)
            if mm:
                m[key] = mm.group(1)
    return m

def split_into_blocks(raw_text: str) -> List[Tuple[int, int]]:
    """
    返回每个块的 [start, end) 区间，块以 STUDY_HEAD 为起点，
    直到下一个 STUDY_HEAD 的起点或文本末尾。
    """
    matches = list(STUDY_HEAD.finditer(raw_text))
    if not matches:
        return []
    bounds = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i+1].start() if i + 1 < len(matches) else len(raw_text)
        bounds.append((start, end))
    return bounds

def parse_20_pairs(block_text: str) -> List[Tuple[str, str]]:
    """
    在块文本中提取紧随其后的 20 行词对（稳妥：在块范围内全局找行，取前20）。
    """
    lines = PAIR_LINE.findall(block_text)
    return lines[:20]

def process_file(input_path: str, output_path: str):
    items = load_json_any(input_path)
    if not items:
        raise SystemExit(f'未读取到任何条目：{input_path}')

    out_records: List[Dict[str, Any]] = []
    global_block_counter = 0

    for src_idx, obj in enumerate(items):
        raw_text = obj.get('text') or pick_longest_string_field(obj)
        if not raw_text:
            continue

        # 元数据优先取对象字段，其次从文本兜底提取
        meta = {
            'participant': obj.get('participant'),
            'experiment': obj.get('experiment'),
            'split': obj.get('split'),
        }
        meta = extract_meta_fallback(raw_text, meta)

        # 找到每个块
        blocks = split_into_blocks(raw_text)
        if not blocks:
            # 没有出现学习标记的样本，跳过
            continue

        for local_block_id, (s, e) in enumerate(blocks):
            block_text = raw_text[s:e]
            study_pairs = parse_20_pairs(block_text)

            # 组织输出小项
            rec = {
                'text': block_text.strip(),
                "experiment": "cox2017information/exp1.csv",
                "participant": obj.get('participant'),
                "split": obj.get('split'),
            }
            out_records.append(rec)
            global_block_counter += 1

    # 写出单一 JSON 文件（数组）
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(out_records, f, ensure_ascii=False, indent=2)

    print(f'完成：切分得到 {len(out_records)} 个块，已写入 {output_path}')

if __name__ == '__main__':
    # 修改为你的路径
    INPUT_PATH = 'cox2017informationexp1csv.json'           # 支持 .json / .jsonl
    OUTPUT_PATH = 'cox2017informationexp1csv_convert.json'
    process_file(INPUT_PATH, OUTPUT_PATH)
