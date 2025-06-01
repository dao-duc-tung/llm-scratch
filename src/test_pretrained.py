import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

# Load the tokenizer
tokenizer_file = "data/fineweb_10k_tokenize_bpe.json"
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)

# Load the model
# model_dir = "models/training/fineweb_10k_gpt2/checkpoint-75"
MODEL_PATH = "models/fineweb_10k_gpt2"
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

input_text = "Vietnam is a"
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
output_ids = model.generate(
    input_ids, max_new_tokens=50, do_sample=True, temperature=0.8, top_k=50, top_p=0.95
)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output_text)
