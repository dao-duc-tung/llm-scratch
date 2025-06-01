from transformers import PreTrainedTokenizerFast


def load_tokenizer_for_instruct(tokenizer_file):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_file,
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
        pad_token="<pad>",
        unk_token="<unk>",
        mask_token="<mask>",
    )
    # use chatML, other format such as alpaca is available
    chat_template = """{% for message in messages -%}
    {{ bos_token }}{{ message['role'] }}
    {{ message['content'] }}{{ eos_token }}
    {% endfor -%}
    {% if add_generation_prompt and messages[-1]['role'] != 'assistant' -%}
    {{ bos_token }}assistant
    {% endif -%}"""
    tokenizer.chat_template = chat_template
    return tokenizer
