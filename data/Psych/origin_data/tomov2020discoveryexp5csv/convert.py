import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ---------- 通用读取 ----------
def load_json_any(path: str) -> List[Dict[str, Any]]:
    """支持 JSON 数组 / 顶层字典 / JSONL"""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
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
        # JSONL 格式
        out = []
        for line in text.splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out
    return []

def pick_longest_string_field(obj: Dict[str, Any]) -> str:
    """挑选对象里最长的字符串字段"""
    best = ""
    for k, v in obj.items():
        if isinstance(v, str) and len(v) > len(best):
            best = v
    return best

# ---------- 切分 ----------
ROUND_HEAD = re.compile(r"The new starting station is", re.M)

def split_rounds_with_rules(raw_text: str) -> Tuple[str, List[str]]:
    """
    返回 (rules_text, round_texts)
    """
    parts = ROUND_HEAD.split(raw_text)
    if len(parts) <= 1:
        return raw_text.strip(), []

    rules_text = parts[0].strip()
    round_texts = ["The new starting station is" + p.strip() for p in parts[1:] if p.strip()]
    return rules_text, round_texts

# ---------- 主流程 ----------
def process_subway_file(input_path: str, output_path: str,
                        default_experiment: str = "subway/navigation"):
    items = load_json_any(input_path)
    if not items:
        raise SystemExit(f"未读取到任何条目：{input_path}")

    out_records: List[Dict[str, Any]] = []

    for obj in items:
        raw_text = obj.get("text") or pick_longest_string_field(obj)
        if not raw_text:
            continue

        meta = {
            "participant": obj.get("participant"),
            "experiment": obj.get("experiment"),
            "split": obj.get("split"),
        }

        rules_text, round_texts = split_rounds_with_rules(raw_text)
        if not round_texts:
            continue

        for round_text in round_texts:
            block_text = f"{rules_text}\n\n{round_text}"

            rec = {
                "text": block_text,
                "experiment": meta.get("experiment") or default_experiment,
                "participant": meta.get("participant"),
                "split": meta.get("split"),
            }
            out_records.append(rec)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_records, f, ensure_ascii=False, indent=2)

    print(f"完成：生成 {len(out_records)} 个 round 项；已写入 {output_path}")

# ---------- 直接运行 ----------
if __name__ == "__main__":
    INPUT_PATH = 'tomov2020discoveryexp5csv.json'          # 输入 JSON / JSONL
    OUTPUT_PATH = 'tomov2020discoveryexp5csv_convert.json'   # 输出 JSON
    process_subway_file(INPUT_PATH, OUTPUT_PATH)
