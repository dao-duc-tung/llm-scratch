from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

# Load the tokenizer
tokenizer_file = "data/fineweb_10k_tokenize_bpe.json"
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)

# Load the model
# model_dir = "models/training/fineweb_10k_gpt2/checkpoint-75"
model_dir = "models/fineweb_10k_gpt2"
model = GPT2LMHeadModel.from_pretrained(model_dir)
# Set the model to evaluation mode
model.eval()

# Prompt for generation
prompt = "Vietnam is a"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Generate text
output = model.generate(
    input_ids,
    # total length (prompt + generated)
    max_length=50,
    num_return_sequences=1,
    # enable sampling for more creative output
    do_sample=True,
    # top-k sampling
    top_k=50,
    # nucleus sampling
    top_p=0.95,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
