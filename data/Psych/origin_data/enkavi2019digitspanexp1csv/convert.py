import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ============ 关键标记与正则 ============
DIGIT_BLOCK_HEAD = re.compile(
    r"You will view a series of digits and are then asked to recall them in the order you have seen them",
    re.M
)
DIGITS_LINE = re.compile(r"The digits are the following:\s*\[([0-9,\s]+)\]", re.M)
PRESS_LINE = re.compile(r"You press <<([^<>]+)>>\.", re.M)

# 如果对象里没有 meta，就尝试从文本中兜底提取（可根据你的数据调整/删除）
META_PATTS = {
    'participant': re.compile(r'"participant"\s*:\s*"([^"]+)"'),
    'experiment':  re.compile(r'"experiment"\s*:\s*"([^"]+)"'),
    'split':       re.compile(r'"split"\s*:\s*"([^"]+)"'),
}

# ============ 通用读取工具 ============
def load_json_any(path: str) -> List[Dict[str, Any]]:
    """支持 JSON 数组 / 顶层带键JSON / JSONL"""
    p = Path(path)
    text = p.read_text(encoding='utf-8')
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    return v
            return [data]
    except json.JSONDecodeError:
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
    """当没有 text 字段时，挑**最长字符串字段**作为原文。"""
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

# ============ 数字任务专用解析 ============
def split_digit_blocks(raw_text: str) -> List[Tuple[int, int]]:
    """
    返回每个数字任务块的 [start, end) 区间，块以 DIGIT_BLOCK_HEAD 为起点，
    直到下一个 DIGIT_BLOCK_HEAD 的起点或文本末尾。
    """
    matches = list(DIGIT_BLOCK_HEAD.finditer(raw_text))
    if not matches:
        return []
    bounds = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i+1].start() if i + 1 < len(matches) else len(raw_text)
        bounds.append((start, end))
    return bounds

def parse_digits_list(block_text: str) -> List[int]:
    """抽取呈现的 digits 序列，如 [4, 8, 2] -> [4,8,2]。"""
    m = DIGITS_LINE.search(block_text)
    if not m:
        return []
    raw = m.group(1)
    nums = []
    for tok in raw.split(','):
        tok = tok.strip()
        if tok.isdigit():
            nums.append(int(tok))
    return nums

def parse_presses_until_S(block_text: str) -> List[str]:
    """抽取按键序列，按出现顺序，包含字母键（可能是 S），保留原样。"""
    presses = [m.group(1).strip() for m in PRESS_LINE.finditer(block_text)]
    return presses

def normalize_press_to_digit(press: str):
    """尝试将 '4' -> 4；非数字（例如 'S'）返回 None。"""
    return int(press) if press.isdigit() else None

def judge_response(digits: List[int], presses: List[str]) -> Dict[str, Any]:
    """
    依据 digits 与按键 presses 评估：
    - response_digits: 遇到第一个 'S' 前的数字键序列
    - terminated_with_S: 是否出现了 'S'（或 's'）
    - is_exact_match: 是否与 digits 完全一致
    - pos_correct: 逐位置是否正确（对齐短序列）
    - correct_prefix_len: 从头开始最长正确前缀长度
    - match_ratio: 逐位置正确的比例（用被试响应长度作为分母；避免奖励多按键）
    """
    # 截断到首个 S/s 之前
    resp = []
    terminated_with_S = False
    for p in presses:
        if p.upper() == 'S':
            terminated_with_S = True
            break
        resp.append(p)

    # 仅保留数字按键
    resp_digits = [normalize_press_to_digit(p) for p in resp]
    resp_digits = [x for x in resp_digits if x is not None]

    # 位置级比较
    L = min(len(digits), len(resp_digits))
    pos_correct = [digits[i] == resp_digits[i] for i in range(L)]

    # 正确前缀长度
    cpre = 0
    for i in range(L):
        if pos_correct[i]:
            cpre += 1
        else:
            break

    is_exact = (resp_digits == digits)
    match_ratio = (sum(pos_correct) / len(resp_digits)) if resp_digits else 0.0

    return {
        'response_digits': resp_digits,
        'terminated_with_S': terminated_with_S,
        'is_exact_match': is_exact,
        'pos_correct': pos_correct,
        'correct_prefix_len': cpre,
        'present_len': len(digits),
        'response_len': len(resp_digits),
        'match_ratio': match_ratio,
    }

# ============ 主流程：批量读取 -> 切块 -> 解析 -> 写JSON ============
def process_digit_file(input_path: str, output_path: str,
                       default_experiment: str = "digit_span/recall"):
    items = load_json_any(input_path)
    if not items:
        raise SystemExit(f'未读取到任何条目：{input_path}')

    out_records: List[Dict[str, Any]] = []
    global_block_counter = 0

    for src_idx, obj in enumerate(items):
        raw_text = obj.get('text') or pick_longest_string_field(obj)
        if not raw_text:
            continue

        # 元信息：优先采用对象字段，其次从文本兜底提取
        meta = {
            'participant': obj.get('participant'),
            'experiment': obj.get('experiment'),
            'split': obj.get('split'),
        }
        meta = extract_meta_fallback(raw_text, meta)

        # 定位每个“数字任务块”
        blocks = split_digit_blocks(raw_text)
        if not blocks:
            # 没有数字任务标记则跳过该对象
            continue

        for local_block_id, (s, e) in enumerate(blocks):
            block_text = raw_text[s:e].strip()
            digits = parse_digits_list(block_text)
            presses = parse_presses_until_S(block_text)

            metrics = judge_response(digits, presses)

            # === 组织输出小项 ===
            # 兼容你之前的字段命名，同时增加结构化字段便于训练/统计
            rec = {
                # 你之前保留的字段
                'text': block_text,
                'experiment': meta.get('experiment') or default_experiment,
                'participant': meta.get('participant'),
                'split': meta.get('split'),

            }

            out_records.append(rec)
            global_block_counter += 1

    # 写出数组 JSON
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(out_records, f, ensure_ascii=False, indent=2)

    print(f'完成：切分并解析 {len(out_records)} 个数字回忆块；已写入 {output_path}')

# ============ 直接运行 ============
if __name__ == '__main__':
    INPUT_PATH = 'enkavi2019digitspanexp1csv.json'           # 你的输入 JSON / JSONL
    OUTPUT_PATH = 'enkavi2019digitspanexp1csv_convert.json'    # 输出数组 JSON
    process_digit_file(INPUT_PATH, OUTPUT_PATH)
