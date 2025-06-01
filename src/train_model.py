from datasets import load_from_disk
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

TOKENIZER_FILE = "data/fineweb_100k_tokenize_bpe.json"
TOKENIZED_DATASET_DIR = "data/tokenized_fineweb_100k"
TRAINING_OUTPUT_DIR = f"models/training/gpt2_01"
MODEL_PATH = f"models/gpt2_01"
CONTEXT_LENGTH = 256

# load the tokenizer
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=TOKENIZER_FILE,
    bos_token="<s>",
    eos_token="</s>",
    pad_token="<pad>",
    unk_token="<unk>",
    mask_token="<mask>",
)


# define the model
# 15e6 rows -> 0.5e9 params
# 10e3 rows -> 0.3e6 params
# 100e3 rows -> 3e6 params, but here we use 1e6 params
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=CONTEXT_LENGTH,
    n_embd=32,
    n_layer=4,
    n_head=4,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)
model = GPT2LMHeadModel(config)

# define the data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=None,
)

# load the tokenized dataset
tokenized_dataset = load_from_disk(TOKENIZED_DATASET_DIR)

# config the training
training_args = TrainingArguments(
    output_dir=TRAINING_OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=4,
    learning_rate=1e-3,
    weight_decay=0.01,
    warmup_ratio=0.03,
    logging_steps=500,
    save_steps=500,
    save_total_limit=4,
    prediction_loss_only=True,
    # no GPU available
    # fp16=True,
    logging_dir=f"{TRAINING_OUTPUT_DIR}/logs",
    # no wandb available
    # report_to="wandb",
    # run_name="fineweb_10k_gpt2",
    # cannot turn on when running on a non-CUDA (CPU or MPS/Apple Silicon) device
    # torch_compile=True,
    lr_scheduler_type="cosine",
    seed=17,
)

# train the model
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
)
trainer.train()
trainer.save_model(MODEL_PATH)
