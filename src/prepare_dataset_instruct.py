import json
import os

from datasets import load_dataset
from tqdm import tqdm

from utils import load_tokenizer_for_instruct

ALPACA_DATASET_NAME = "tatsu-lab/alpaca"
DOLLY_DATASET_NAME = "databricks/databricks-dolly-15k"
SAMPLE_SIZE = 500
TRAIN_CONVOS_FILE = f"data/alpaca_{SAMPLE_SIZE}_dolly_{SAMPLE_SIZE}/train_convos.jsonl"
VAL_CONVOS_FILE = f"data/alpaca_{SAMPLE_SIZE}_dolly_{SAMPLE_SIZE}/val_convos.jsonl"
SEED = 17
MAX_LENGTH = 1024


TOKENIZER_FILE = "data/fineweb_10k_tokenize_bpe.json"
tokenizer = load_tokenizer_for_instruct(TOKENIZER_FILE)

print("Loading ALPACA")
alpaca_split = load_dataset(ALPACA_DATASET_NAME, split="train")
alpaca_split = alpaca_split.select(range(SAMPLE_SIZE)).train_test_split(
    test_size=0.1, seed=SEED, shuffle=True
)
alpaca_train = alpaca_split["train"]
alpaca_val = alpaca_split["test"]


def format_alpaca(ds_split):
    # convos = conversations
    convos = []
    for ex in ds_split:
        instr = ex.get("instruction", "").strip()
        ctx = ex.get("input", "").strip()
        response = ex.get("output", "").strip()

        user_msg_parts = [instr]
        # Add context only if it's not empty
        if ctx:
            user_msg_parts.append(ctx)
        user_msg = "\n\n".join(user_msg_parts)

        if user_msg and response:
            convos.append(
                [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": response},
                ]
            )
    return convos


train_alpaca_convos = format_alpaca(alpaca_train)
val_alpaca_convos = format_alpaca(alpaca_val)
print(
    f"Alpaca -> {len(train_alpaca_convos)} train / {len(val_alpaca_convos)} val convos"
)

print("Loading Dolly 15k")
dolly = load_dataset(DOLLY_DATASET_NAME, split="train")
dolly = dolly.select(range(SAMPLE_SIZE))

# Filter out examples with missing 'instruction' or 'response' in Dolly
dolly_filtered = dolly.filter(
    lambda ex: ex.get("instruction")
    and ex.get("response")
    and len(ex["instruction"].strip()) > 0
    and len(ex["response"].strip()) > 0
)

dolly_split = dolly_filtered.train_test_split(test_size=0.1, seed=SEED, shuffle=True)
dolly_train = dolly_split["train"]
dolly_val = dolly_split["test"]


def format_dolly(ds_split):
    convos = []
    for ex in ds_split:
        instr = ex.get("instruction", "").strip()
        ctx = ex.get("context", "").strip()
        response = ex.get("response", "").strip()

        user_msg_parts = [instr]
        if ctx:
            user_msg_parts.append(ctx)
        user_msg = "\n\n".join(user_msg_parts)

        if user_msg and response:
            convos.append(
                [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": response},
                ]
            )
    return convos


train_dolly_convos = format_dolly(dolly_train)
val_dolly_convos = format_dolly(dolly_val)
print(
    f"Dolly-15k -> {len(train_dolly_convos)} train / {len(val_dolly_convos)} val convos"
)

# combine the 2 datasets
all_train_convos = train_alpaca_convos + train_dolly_convos
all_val_convos = val_alpaca_convos + val_dolly_convos
print(
    f"Combined (raw) -> {len(all_train_convos)} train / {len(all_val_convos)} val convos"
)


# remove those > context length
def filter_and_prepare_conversations(convos, tokenizer_ref, max_len):
    filtered_convos = []
    for convo in tqdm(convos, desc="Filtering long conversations"):
        # skip empty conversation
        if not convo:
            continue
        prompt_text = tokenizer_ref.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False
        )
        tokenized_len = len(tokenizer_ref.encode(prompt_text, truncation=False))

        if tokenized_len > 0 and tokenized_len <= max_len:
            filtered_convos.append({"conversations": convo})
        elif tokenized_len == 0:
            print(f"Empty conversation {convo}")

    return filtered_convos


train_convos_filtered = filter_and_prepare_conversations(
    all_train_convos, tokenizer, MAX_LENGTH
)
val_convos_filtered = filter_and_prepare_conversations(
    all_val_convos, tokenizer, MAX_LENGTH
)

print(
    f"Filtered by length -> {len(train_convos_filtered)} train / {len(val_convos_filtered)} val convos"
)


def save_conversations_to_jsonl(convos, file_path: str):
    parent = file_path.rsplit("/", 1)[0]
    if not os.path.exists(parent):
        os.makedirs(parent)

    with open(file_path, "w", encoding="utf-8") as f:
        for row in convos:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")


save_conversations_to_jsonl(train_convos_filtered, TRAIN_CONVOS_FILE)
save_conversations_to_jsonl(val_convos_filtered, VAL_CONVOS_FILE)
