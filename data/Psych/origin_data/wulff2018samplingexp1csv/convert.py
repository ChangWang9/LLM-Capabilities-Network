import json
import re
from pathlib import Path

def split_blocks(raw_text: str):
    # 切分点：遇到 "You encounter a new choice problem:"
    parts = re.split(r"You encounter a new choice problem:", raw_text)
    blocks = []

    # 统一规则头
    rules_header = (
        "You can sample from two monetary lotteries by pressing K or D.\n"
        "The lotteries offer different points with different probabilities.\n"
        "Initially, you will not know the outcomes and probabilities of the lotteries, but you can learn about them through sampling.\n"
        "Whenever you sample, a random draw from the selected lottery will be generated, which does not affect your bonus.\n"
        "You can sample from the lotteries in whatever order and for as long as you like.\n"
        "Whenever you feel ready, you can stop sampling by pressing X and then choose one lottery for real by pressing the corresponding key.\n"
        "This choice will then trigger a random draw from the chosen lottery that will be added to your bonus.\n"
        "Your goal is to maximize your bonus.\n"
        "You will be presented with multiple choice problems consisting of different lotteries varying in outcomes and probabilities."
    )

    # parts[0] 是规则文本，可以直接忽略（因为 rules_header 已经写死）
    for i in range(1, len(parts)):
        body = parts[i].strip()
        if body:  # 跳过空的
            blocks.append(rules_header + "\n\nYou encounter a new choice problem:\n" + body)

    return blocks

def process_json_file(input_path: str, output_path: str,
                      default_experiment="lottery_sampling",
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
    INPUT_PATH = 'wulff2018samplingexp1csv.json'   # 输入 JSON 文件
    OUTPUT_PATH = 'wulff2018samplingexp1csv_convert.json'  # 输出 JSON 文件
    process_json_file(INPUT_PATH, OUTPUT_PATH)
