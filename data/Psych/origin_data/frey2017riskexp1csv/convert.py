import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ---------- 通用读取 ----------
def load_json_any(path: str) -> List[Dict[str, Any]]:
    """支持 JSON 数组 / 顶层带键 JSON / JSONL"""
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
    """无 text 字段时，取最长的字符串字段作为原文。"""
    best = ''
    for k, v in obj.items():
        if isinstance(v, str) and len(v) > len(best):
            best = v
    return best

# 若对象缺 meta，则从原文中兜底提取（可按需删除）
META_PATTS = {
    'participant': re.compile(r'"participant"\s*:\s*"([^"]+)"'),
    'experiment':  re.compile(r'"experiment"\s*:\s*"([^"]+)"'),
    'split':       re.compile(r'"split"\s*:\s*"([^"]+)"'),
}
def extract_meta_fallback(raw_text: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    m = dict(meta)
    for key, patt in META_PATTS.items():
        if not m.get(key):
            mm = patt.search(raw_text)
            if mm:
                m[key] = mm.group(1)
    return m

# ---------- 解析“规则 + Balloon n” ----------
BALLOON_HEAD = re.compile(r'^Balloon\s+(\d+):', re.M)

def split_balloons_with_rules(raw_text: str) -> Tuple[str, List[Tuple[int, int]]]:
    """
    返回 (rules_text, balloon_spans)
    rules_text: 从文本开头到 "Balloon 1:" 之前
    balloon_spans: 每个 Balloon 的 [start, end) 区间
    """
    matches = list(BALLOON_HEAD.finditer(raw_text))
    if not matches:
        return raw_text.strip(), []

    rules_text = raw_text[:matches[0].start()].strip()

    spans = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i+1].start() if i + 1 < len(matches) else len(raw_text)
        spans.append((start, end))
    return rules_text, spans

# ---------- 主流程 ----------
def process_balloon_file(input_path: str, output_path: str,
                         default_experiment: str = "balloon_risk/balloons"):
    items = load_json_any(input_path)
    if not items:
        raise SystemExit(f'未读取到任何条目：{input_path}')

    out_records: List[Dict[str, Any]] = []

    for obj in items:
        raw_text = obj.get('text') or pick_longest_string_field(obj)
        if not raw_text:
            continue

        # 元信息
        meta = {
            'participant': obj.get('participant'),
            'experiment': obj.get('experiment'),
            'split': obj.get('split'),
        }
        meta = extract_meta_fallback(raw_text, meta)

        # 切分规则 + 每个 Balloon
        rules_text, balloon_spans = split_balloons_with_rules(raw_text)
        if not balloon_spans:
            continue

        for (s, e) in balloon_spans:
            balloon_text = raw_text[s:e].strip()
            block_text = (rules_text + "\n\n" + balloon_text).strip()

            rec = {
                'text': block_text,
                'experiment': meta.get('experiment') or default_experiment,
                'participant': meta.get('participant'),
                'split': meta.get('split'),
            }
            out_records.append(rec)

    # 写出数组 JSON
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(out_records, f, ensure_ascii=False, indent=2)

    print(f'完成：生成 {len(out_records)} 个 Balloon 项；已写入 {output_path}')

# ---------- 直接运行 ----------
if __name__ == '__main__':
    INPUT_PATH = 'frey2017riskexp1csv.json'           # 你的输入 JSON / JSONL
    OUTPUT_PATH = 'frey2017riskexp1csv_convert.json'    # 输出数组 JSON
    process_balloon_file(INPUT_PATH, OUTPUT_PATH)
