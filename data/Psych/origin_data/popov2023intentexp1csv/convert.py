import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

# -------- 读取工具 --------
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
        out = []
        for line in text.splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out
    return []

def pick_longest_string_field(obj: Dict[str, Any]) -> str:
    best = ""
    for k, v in obj.items():
        if isinstance(v, str) and len(v) > len(best):
            best = v
    return best

# -------- 解析 --------
LIST_HEAD = re.compile(r"^List\s+(\d+),\s*task\s*1:", re.M)

def split_lists_with_rules(raw_text: str) -> Tuple[str, List[Tuple[int, int, str]]]:
    """
    返回 (rules_text, list_spans)
    list_spans: (start, end, list_id)
    """
    matches = list(LIST_HEAD.finditer(raw_text))
    if not matches:
        return raw_text.strip(), []

    rules_text = raw_text[:matches[0].start()].strip()

    spans = []
    for i, m in enumerate(matches):
        list_id = m.group(1)
        start = m.start()
        end = matches[i+1].start() if i + 1 < len(matches) else len(raw_text)
        spans.append((start, end, list_id))
    return rules_text, spans

# -------- 主流程 --------
def process_memory_task_file(input_path: str, output_path: str,
                             default_experiment: str = "memory/words_arithmetic"):
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

        # 切分规则 + List
        rules_text, list_spans = split_lists_with_rules(raw_text)
        if not list_spans:
            continue

        for (s, e, list_id) in list_spans:
            list_text = raw_text[s:e].strip()
            block_text = f"{rules_text}\n\n{list_text}"

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

    print(f"完成：生成 {len(out_records)} 个 List 项；已写入 {output_path}")

# -------- 直接运行 --------
if __name__ == "__main__":
    INPUT_PATH = 'popov2023intentexp1csv.json'          # 输入 JSON / JSONL
    OUTPUT_PATH = 'popov2023intentexp1csv_convert.json'   # 输出 JSON
    process_memory_task_file(INPUT_PATH, OUTPUT_PATH)
