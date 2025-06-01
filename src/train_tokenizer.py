from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

DATA_PATH = "data/fineweb_100k.jsonl"
OUTPUT_PATH = "data/fineweb_100k_tokenize_bpe.json"

dataset = load_dataset("json", data_files=DATA_PATH, split="train")
dataset = dataset.remove_columns([col for col in dataset.column_names if col != "text"])

# train a byte-level BPE tokenizer
tokenizer = ByteLevelBPETokenizer()


def batch_iterator(dataset, batch_size=500):
    for batch in dataset.iter(batch_size=batch_size):
        yield batch["text"]


special_tokens = ["<s>", "</s>", "<pad>", "<unk>", "<mask>"]


tokenizer.train_from_iterator(
    batch_iterator(dataset),
    vocab_size=50_000,
    min_frequency=2,
    special_tokens=special_tokens,
    show_progress=True,
)

tokenizer.save(OUTPUT_PATH)
