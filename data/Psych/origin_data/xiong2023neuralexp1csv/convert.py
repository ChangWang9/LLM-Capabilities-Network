import json
import re
from pathlib import Path

def split_blocks(raw_text: str):
    # 切分点：每个 Game 开头
    parts = re.split(r"(Game \d+\..*?game\.)", raw_text, flags=re.DOTALL)
    blocks = []

    # 规则头部（固定保留）
    rules_header = (
        "You are participating in multiple games involving two slot machines, labeled M and V.\n"
        "The two slot machines are different in different games.\n"
        "Each time you choose a slot machine, you get points (choosing the same slot machine will not always give you the same points).\n"
        "You select a slot machine by pressing the corresponding key.\n"
        "The expected points change randomly, abruptly, and independently with a hazard rate (which you will be told).\n"
        "When the points change, the new expected point value assigned to that slot machine is sampled from a uniform distribution (from 1 to 99 points).\n"
        "For example, if the hazard rate is 0.1, the expected points of the machines change with 10%.\n"
        "Your goal is to choose the slot machine that will give you the most points."
    )

    # parts: ['', 'Game 1. ... game.', '内容', 'Game 2. ... game.', '内容', ...]
    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        body = parts[i+1].strip() if i+1 < len(parts) else ""
        if header:
            blocks.append(rules_header + "\n\n" + header + "\n" + body)

    return blocks

def process_json_file(input_path: str, output_path: str,
                      default_experiment="slot_machine_hazard",
                      participant="0", split="train"):
    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    out_records = []

    for item in data:
        raw_text = item["text"]
        blocks = split_blocks(raw_text)

        for block_text in blocks:
            rec = {
                "text": block_text,
                "experiment": default_experiment,
                "participant": item.get("participant", participant),
                "split": item.get("split", split)
            }
            out_records.append(rec)

    Path(output_path).write_text(
        json.dumps(out_records, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"完成：共生成 {len(out_records)} 条记录，写入 {output_path}")

if __name__ == "__main__":
    INPUT_PATH = 'xiong2023neuralexp1csv.json'   # 输入 JSON 文件
    OUTPUT_PATH = 'xiong2023neuralexp1csv_convert.json'  # 输出 JSON 文件
    process_json_file(INPUT_PATH, OUTPUT_PATH)
