import types

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from utils import load_tokenizer_for_instruct

MODEL_PATH = "models/gpt2_01"
TOKENIZER_FILE = "data/fineweb_100k_tokenize_bpe.json"
OUTPUT_DIR = "models/instruct/gpt2_01_alpaca_dolly_neftune"
ALPACA_SAMPLE_SIZE = 50000
DOLLY_SAMPLE_SIZE = 10000
TRAIN_CONVOS_FILE = (
    f"data/alpaca_{ALPACA_SAMPLE_SIZE}_dolly_{DOLLY_SAMPLE_SIZE}/train_convos.jsonl"
)
VAL_CONVOS_FILE = (
    f"data/alpaca_{ALPACA_SAMPLE_SIZE}_dolly_{DOLLY_SAMPLE_SIZE}/val_convos.jsonl"
)
FINAL_MODEL_PATH = "models/gpt2_01_instruct"
PRETRAINED_TOKENIZER_PATH = f"{FINAL_MODEL_PATH}/pretrained_tokenizer"

MAX_LENGTH = 256
NEFTUNE_NOISE_ALPHA = 5.0


print("Loading tokenizer")
tokenizer = load_tokenizer_for_instruct(TOKENIZER_FILE)

print("Loading model")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
model.resize_token_embeddings(len(tokenizer))
model.config.bos_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

print(
    f"Tokenizer BOS: {tokenizer.bos_token} ({tokenizer.bos_token_id}), EOS: {tokenizer.eos_token} ({tokenizer.eos_token_id}), PAD: {tokenizer.pad_token} ({tokenizer.pad_token_id})"
)
print(
    f"Model BOS: {model.config.bos_token_id}, EOS: {model.config.eos_token_id}, PAD: {model.config.pad_token_id}"
)
assert model.config.pad_token_id is not None, "Model's pad_token_id is not set!"


train_convos = load_dataset("json", data_files=TRAIN_CONVOS_FILE, split="train")
val_convos = load_dataset("json", data_files=VAL_CONVOS_FILE, split="train")
train_dataset = Dataset.from_list(train_convos)
val_dataset = Dataset.from_list(val_convos)


ASSISTANT_ROLE_NAME = "assistant"
USER_ROLE_NAME = "user"


def formatting_func(example):
    # 'example' is something like {"conversations": [{"role": ..., "content": ...}, ...]}
    conversation = example["conversations"]

    prompt_text = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False
    )

    tokenized_inputs = tokenizer(
        prompt_text,
        # not really needed here bcs we already remove those that > max length
        truncation=True,
        max_length=MAX_LENGTH,
        return_attention_mask=True,
        padding="max_length",
    )
    input_ids = tokenized_inputs["input_ids"]

    labels = [-100] * len(input_ids)

    # find the last assistant response
    last_assistant_idx = max(
        idx
        for idx, turn in enumerate(conversation)
        if turn["role"] == ASSISTANT_ROLE_NAME
    )

    # Iterate through the tokenized input_ids to unmask assistant content

    current_token_idx = 0
    for turn_idx, turn in enumerate(conversation):
        role = turn["role"]
        content = turn["content"]

        try:
            start_of_turn_bos_idx = input_ids.index(
                tokenizer.bos_token_id, current_token_idx
            )
        except ValueError:
            break

        # Search after the current BOS
        search_for_eos_from = start_of_turn_bos_idx + 1
        end_of_turn_eos_idx = -1

        for k_eos in range(search_for_eos_from, len(input_ids)):
            if input_ids[k_eos] == tokenizer.eos_token_id:
                end_of_turn_eos_idx = k_eos
                break
        if end_of_turn_eos_idx == -1:
            print(f"Warning: Could not find EOS token for turn: {turn}")
            return None

        role_and_newline_text = f"{role}\n"
        role_and_newline_tokens = tokenizer.encode(
            role_and_newline_text, add_special_tokens=False
        )

        start_of_content_idx = (
            start_of_turn_bos_idx + 1 + len(role_and_newline_tokens)
        )  # +1 for BOS

        if role == ASSISTANT_ROLE_NAME and turn_idx == last_assistant_idx:
            # Unmask tokens from start_of_content_idx up to (but not including) end_of_turn_eos_idx
            for k_label in range(start_of_content_idx, end_of_turn_eos_idx):
                if k_label >= 0 and k_label < len(labels):
                    labels[k_label] = input_ids[k_label]

        current_token_idx = (
            end_of_turn_eos_idx + 1
        )  # Move search for next turn after this turn's EOS

    return {
        "input_ids": input_ids,
        "attention_mask": tokenized_inputs["attention_mask"],
        "labels": labels,
    }


tokenized_train = train_dataset.map(formatting_func)
tokenized_val = val_dataset.map(formatting_func)

# Set format
tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_val.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# NEFTune adds noise to the embedding vectors during training. Standard finetuning of LLaMA-2-7B using Alpaca achieves 29.79% on AlpacaEval, which rises to 64.69% using noisy embeddings.
def add_neftune(embedding_layer, noise_alpha):
    original_forward = embedding_layer.forward

    def new_forward(self, input_ids):
        if self.training:
            embed_init = original_forward(input_ids)
            L = input_ids.size(1)
            d = embed_init.size(2)
            mag_norm = noise_alpha / (L * d) ** 0.5
            # generate uniform noise
            noise = torch.zeros_like(embed_init).uniform_(-mag_norm, mag_norm)
            return embed_init + noise
        else:
            # during inference, return standard embeddings without noise
            return original_forward(input_ids)

    # bind the new forward method to the embedding layer
    embedding_layer.forward = types.MethodType(new_forward, embedding_layer)


embedding_layer = model.get_input_embeddings()
add_neftune(embedding_layer, noise_alpha=NEFTUNE_NOISE_ALPHA)

early_stopping = EarlyStoppingCallback(early_stopping_patience=3)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=4,
    save_steps=500,
    save_total_limit=6,
    logging_steps=50,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="linear",
    # fp16=True,
    eval_strategy="steps",
    eval_steps=500,
    load_best_model_at_end=True,
    logging_dir=f"{OUTPUT_DIR}/logs",
    remove_unused_columns=False,
    # torch_compile=True,
    seed=17,
    # report_to="wandb",
    # run_name="gpt2-chat-0p3",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
    tokenizer=tokenizer,
    callbacks=[early_stopping],
)

print("Starting fine-tuning")
trainer.train()

trainer.save_model(FINAL_MODEL_PATH)
tokenizer.save_pretrained(PRETRAINED_TOKENIZER_PATH)
