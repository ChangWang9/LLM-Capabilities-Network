import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

def split_games(raw_text: str) -> Tuple[str, List[str]]:
    """拆分规则说明和每个Game内容"""
    parts = re.split(r"(Game \d+\..*?)(?=Game \d+\.|$)", raw_text, flags=re.S)
    rules = parts[0].strip()
    games = []
    for i in range(1, len(parts), 2):
        game_header = parts[i].strip()
        game_body = parts[i+1].strip() if i+1 < len(parts) else ""
        games.append(f"{game_header}\n{game_body}".strip())
    return rules, games

def process_json_file(input_path: str, output_path: str,
                      default_experiment: str = "slot_machine/CA",
                      participant: str = "unknown",
                      split: str = "train"):
    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    out_records: List[Dict[str, Any]] = []

    for item in data:
        raw_text = item["text"]
        rules, game_texts = split_games(raw_text)

        for game_id, game in enumerate(game_texts, start=1):
            rec = {
                "game_id": game_id,
                "text": f"{rules}\n\n{game}",
                "experiment": default_experiment,
                "participant": item.get("participant", participant),
                "split": item.get("split", split),
                "total_games": len(game_texts)
            }
            out_records.append(rec)

    Path(output_path).write_text(json.dumps(out_records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成：生成 {len(out_records)} 个 game 条目，已写入 {output_path}")

if __name__ == "__main__":
    INPUT_PATH = 'wilson2014humansexp1csv.json'          # 输入 JSON
    OUTPUT_PATH = 'wilson2014humansexp1csv_convert.json'   # 输出 JSON
    process_json_file(INPUT_PATH, OUTPUT_PATH)
