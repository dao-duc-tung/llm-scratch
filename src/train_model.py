from datasets import load_from_disk
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

# load the tokenizer
tokenizer_file = "data/fineweb_10k_tokenize_bpe.json"
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=tokenizer_file,
    bos_token="<s>",
    eos_token="</s>",
    pad_token="<pad>",
    unk_token="<unk>",
    mask_token="<mask>",
)


# define the model
# 15e6 rows -> 0.5e9 params
# 10e3 rows -> 0.3e6 params
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=1024,
    n_embd=8,
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
tokenized_dataset_dir = "data/tokenized_fineweb_10k"
tokenized_dataset = load_from_disk(tokenized_dataset_dir)

# config the training
training_args = TrainingArguments(
    output_dir="models/training/fineweb_10k_gpt2",
    overwrite_output_dir=True,
    num_train_epochs=1,
    # effective batch size = 16*4 = 64
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    learning_rate=1e-3,
    weight_decay=0.01,
    warmup_ratio=0.03,
    logging_steps=25,
    save_steps=25,
    save_total_limit=4,
    prediction_loss_only=True,
    # no GPU available
    # fp16=True,
    logging_dir="logs",
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
trainer.save_model("models/fineweb_10k_gpt2")
