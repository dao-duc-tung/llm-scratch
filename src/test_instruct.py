import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

MODEL_PATH = "models/gpt2_01"
FINAL_MODEL_PATH = "models/gpt2_01_instruct"
PRETRAINED_TOKENIZER_PATH = f"{FINAL_MODEL_PATH}/pretrained_tokenizer"

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_TOKENIZER_PATH)

print("Testing pretrained model")
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


print("Testing instructed model")
pipe = pipeline(
    "text-generation",
    model=FINAL_MODEL_PATH,
    tokenizer=FINAL_MODEL_PATH,
    device=0 if torch.cuda.is_available() else -1,
)
messages = [
    {"role": "user", "content": "Complete this sentence: Vietnam is a"},
]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
outputs = pipe(
    prompt,
    max_new_tokens=50,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    repetition_penalty=1.1,
)
print(outputs[0]["generated_text"])
