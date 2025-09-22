import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ---------- 读取 ----------
def load_json_any(path: str) -> List[Dict[str, Any]]:
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
BOARD_HEAD = re.compile(r"You are currently solving board\s+(\d+)\.")
STATE_HEAD = re.compile(r"The board state is now:")

def split_states_with_rules(raw_text: str) -> Tuple[str, List[Tuple[int, int, str]]]:
    """
    返回 (rules_text, state_spans)
    state_spans: 每个 state 的 [start, end, board_id]
    """
    # 找 board 起点
    boards = list(BOARD_HEAD.finditer(raw_text))
    if not boards:
        return raw_text.strip(), []

    rules_text = raw_text[:boards[0].start()].strip()
    spans = []

    for bi, bm in enumerate(boards):
        board_id = bm.group(1)
        b_start = bm.end()
        b_end = boards[bi+1].start() if bi + 1 < len(boards) else len(raw_text)
        board_text = raw_text[b_start:b_end]

        # 找 state 起点
        states = list(STATE_HEAD.finditer(board_text))
        for si, sm in enumerate(states):
            s_start = sm.start()
            s_end = states[si+1].start() if si + 1 < len(states) else len(board_text)
            spans.append((b_start + s_start, b_start + s_end, board_id))
    return rules_text, spans

# ---------- 主流程 ----------
def process_binary_grid_states(input_path: str, output_path: str,
                               default_experiment: str = "binary_grid/reveal"):
    items = load_json_any(input_path)
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

        rules_text, state_spans = split_states_with_rules(raw_text)
        for (s, e, board_id) in state_spans:
            state_text = raw_text[s:e].strip()
            block_text = f"{rules_text}\n\nYou are currently solving board {board_id}.\n{state_text}"

            rec = {
                "text": block_text.strip(),
                "experiment": meta.get("experiment") or default_experiment,
                "participant": meta.get("participant"),
                "split": meta.get("split"),
            }
            out_records.append(rec)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_records, f, ensure_ascii=False, indent=2)

    print(f"完成：生成 {len(out_records)} 个 state 项；已写入 {output_path}")

# ---------- 直接运行 ----------
if __name__ == "__main__":
    INPUT_PATH = 'kumar2023disentanglingexp1csv.json'            # 输入文件
    OUTPUT_PATH = 'kumar2023disentanglingexp1csv_convert.json'    # 输出文件
    process_binary_grid_states(INPUT_PATH, OUTPUT_PATH)
