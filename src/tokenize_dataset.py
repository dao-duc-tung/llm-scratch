import os

from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="data/fineweb_10k_tokenize_bpe.json",
    bos_token="<s>",
    eos_token="</s>",
    pad_token="<pad>",
    unk_token="<unk>",
    mask_token="<mask>",
)

data_path = "data/fineweb_10k.jsonl"
dataset = load_dataset("json", data_files=data_path, split="train")
dataset = dataset.remove_columns([col for col in dataset.column_names if col != "text"])

context_length = 1024


def tokenize_function(batch):
    # Concatenate all texts in the batch, add EOS after each
    all_token_ids = []
    for text in batch["text"]:
        all_token_ids.extend(tokenizer.encode(text, add_special_tokens=False))
        all_token_ids.append(tokenizer.eos_token_id)

    # Split into fixed-size chunks, ignore incomplete chunks
    # example of values of chunks
    # [
    #     [123, 456, 789, ..., 42],  # 1024 integers
    #     [234, 567, 890, ..., 99],  # 1024 integers
    #     ...
    # ]
    chunks = []
    for i in range(0, len(all_token_ids), context_length):
        chunk = all_token_ids[i : i + context_length]
        if len(chunk) == context_length:
            chunks.append(chunk)

    # Hugging Face Datasets’ map method requires to return a dict mapping new column names to values (eg. {"input_ids": ...})
    return {"input_ids": chunks, "labels": chunks.copy()}


tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    batch_size=1000,
    remove_columns=dataset.column_names,
    num_proc=os.cpu_count(),
)

tokenized_dataset_dir = "data/tokenized_fineweb_10k"
tokenized_dataset.save_to_disk(tokenized_dataset_dir)

print(tokenized_dataset[0])
