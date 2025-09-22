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
        # JSONL
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

# ---------- 解析 ----------
OPTION_HEAD = re.compile(r"Option F delivers", re.M)

def split_problems_with_rules(raw_text: str) -> Tuple[str, List[Tuple[int, int]]]:
    """
    返回 (rules_text, problem_spans)
    rules_text: 开头到第一个 Option 前
    problem_spans: 每个 problem 的 [start, end)
    """
    matches = list(OPTION_HEAD.finditer(raw_text))
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
def process_gambling_feedback_file(input_path: str, output_path: str,
                                   default_experiment: str = "gambling/feedback"):
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

        # 切分规则 + problems
        rules_text, problem_spans = split_problems_with_rules(raw_text)
        if not problem_spans:
            continue

        for (s, e) in problem_spans:
            prob_text = raw_text[s:e].strip()
            block_text = f"{rules_text}\n\n{prob_text}"

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

    print(f"完成：生成 {len(out_records)} 个 gambling problem 项；已写入 {output_path}")

# ---------- 直接运行 ----------
if __name__ == "__main__":
    INPUT_PATH = 'plonsky2018whenexp1csv.json'          # 输入 JSON / JSONL
    OUTPUT_PATH = 'plonsky2018whenexp1csv_convert.json'   # 输出 JSON
    process_gambling_feedback_file(INPUT_PATH, OUTPUT_PATH)
