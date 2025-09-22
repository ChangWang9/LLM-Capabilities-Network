import json
import re
from pathlib import Path

def split_blocks(raw_text: str):
    # 切分点：遇到 "Act as fast and accurately as possible."
    parts = re.split(r"(Act as fast and accurately as possible\.)", raw_text)
    blocks = []

    # 统一规则头
    rules_header = "Press the instructed key.\nAct as fast and accurately as possible."

    if parts:
        # 第一个 block
        if len(parts) >= 3:
            body = parts[2].strip()
            blocks.append(rules_header + "\n\n" + body)

        # 后续 block
        for i in range(3, len(parts), 2):
            body = parts[i].strip()
            blocks.append(rules_header + "\n\n" + body)

    return blocks

def process_json_file(input_path: str, output_path: str,
                      default_experiment="reaction_time",
                      participant="unknown", split="train"):
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

    Path(output_path).write_text(json.dumps(out_records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成：共生成 {len(out_records)} 条记录，写入 {output_path}")

if __name__ == "__main__":
    INPUT_PATH = 'wu2023chunkingexp2csv.json'   # 输入 JSON 文件
    OUTPUT_PATH = 'wu2023chunkingexp2csv_convert.json'  # 输出 JSON 文件
    process_json_file(INPUT_PATH, OUTPUT_PATH)
