import json
import re
from pathlib import Path
from typing import List, Dict

def split_blocks(raw_text: str, rules: str) -> List[str]:
    """按 break 标记切分成多个 block，并保留最开头的规则"""
    blocks = re.split(
        r"There is a short break before the next block\. The probability of electric shocks associated with the stimuli will be reset\. Please disregard anything you have learned in the previous block\.",
        raw_text
    )
    # 去掉空行和首尾空格
    blocks = [b.strip() for b in blocks if b.strip()]
    return [rules + "\n\n" + b for b in blocks]

def process_json_file(input_path: str, output_path: str,
                      default_experiment: str = "shock_prediction",
                      participant: str = "unknown",
                      split: str = "train"):
    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    out_records: List[Dict] = []

    for item in data:
        raw_text = item["text"]

        # 提取规则部分（最开头到第一个 "Stimulus" 之前）
        match = re.search(r"(.*?)Stimulus", raw_text, re.S)
        if match:
            rules = match.group(1).strip()
        else:
            rules = "No explicit rules found"

        blocks = split_blocks(raw_text, rules)

        for block_id, block_text in enumerate(blocks, start=1):
            rec = {
                "text": block_text,
                "experiment": default_experiment,
                "participant": item.get("participant", participant),
                "split": item.get("split", split),
                
                }
            out_records.append(rec)

    Path(output_path).write_text(json.dumps(out_records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成：共生成 {len(out_records)} 个 block，已写入 {output_path}")

if __name__ == "__main__":
    INPUT_PATH = 'wise2019acomputationalexp1csv.json'         # 输入 JSON 文件
    OUTPUT_PATH = 'wise2019acomputationalexp1csv_convert.json'        # 输出 JSON 文件
    process_json_file(INPUT_PATH, OUTPUT_PATH)
