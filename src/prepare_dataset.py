import json

from datasets import load_dataset

dataset = load_dataset(
    "HuggingFaceFW/fineweb",
    "sample-10BT",
    split="train",
    streaming=False,
)

# use only 'text' column to speed up training
# ref: https://discuss.huggingface.co/t/speed-issues-using-tokenizer-train-new-from-iterator-on-50gb-dataset/29125
dataset = dataset.remove_columns([col for col in dataset.column_names if col != "text"])

# Extract sample rows
sampled_dataset = dataset.select(range(10000))
output_path = "data/fineweb_10k.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for row in sampled_dataset:
        json.dump(row, f, ensure_ascii=False)
        f.write("\n")
