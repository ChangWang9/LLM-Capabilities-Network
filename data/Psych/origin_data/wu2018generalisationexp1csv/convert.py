import json
import re
from pathlib import Path
from typing import List, Dict

def split_environments(raw_text: str, rules: str) -> List[str]:
    """按 Environment number 切分，并在每个 block 前加上规则"""
    parts = re.split(r"(Environment number \d+:)", raw_text)
    blocks = []
    # parts: ['', 'Environment number 1:', '内容1', 'Environment number 2:', '内容2', ...]
    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        body = parts[i+1].strip()
        block_text = rules + "\n\n" + header + "\n" + body
        blocks.append(block_text)
    return blocks

def process_json_file(input_path: str, output_path: str,
                      default_experiment: str = "environment_explore",
                      participant: str = "unknown",
                      split: str = "train"):
    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    out_records: List[Dict] = []

    for item in data:
        raw_text = item["text"]

        # 规则：最开头到 "Environment number" 前
        match = re.search(r"(.*?)Environment number", raw_text, re.S)
        rules = match.group(1).strip() if match else "No explicit rules found"

        blocks = split_environments(raw_text, rules)

        for env_id, block_text in enumerate(blocks, start=1):
            rec = {
            
                "text": block_text,
                "experiment": default_experiment,
                "participant": item.get("participant", participant),
                "split": item.get("split", split),
                
            }
            out_records.append(rec)

    Path(output_path).write_text(json.dumps(out_records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成：共生成 {len(out_records)} 个 environment，已写入 {output_path}")

if __name__ == "__main__":
    INPUT_PATH = 'wu2018generalisationexp1csv.json'   # 输入 JSON 文件
    OUTPUT_PATH = 'wu2018generalisationexp1csv_convert.json'  # 输出 JSON 文件
    process_json_file(INPUT_PATH, OUTPUT_PATH)
