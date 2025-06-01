# LLM from scratch

The goal is to build an LLM from scratch that does:

- Construct sentences in English without major grammatical flaws
- Generate a coherent response to my query
- No need to be factually correct
- Not too large in param size, less than 1B params
- 1024 context length

Setup environment:

```bash
pyenv virtualenv 3.13.3 llm
pyenv local llm
```

## 1. Prepare a dataset

The file `src/prepare_dataset.py` downloads the dataset and creates a sample dataset of 10k rows.

### 1.1. Why do we need a dataset?

The dataset provides examples for the model to learn language patterns such as:

- Learn grammar: the model observes correct English grammar in the dataset and learns to reproduce it
- Build vocabulary: the model observes a wide range of words and phrases
- Understand context: the model observes how sentences and ideas flow and learns to generate coherent responses

### 1.2. What are criteria to consider a dataset "good enough"?

- Quality
  - Contain well-formed, grammatical English sentences
  - Minimal noise (eg. code, non-English text, spam)
- Diversity
  - Cover a wide range of topics, writing styles, and sentence structures
  - Include both formal and informal language
- Size:
  - Large enough to expose the model to varied patterns (hundreds of millions to billions of tokens is typical, but smaller is possible for experiments)
- Relevance
  - Match our target use case (eg. conversational data for chatbots, Wikipedia for factual writing)
- Cleanliness
  - Duplicates, corrupted files, and irrelevant content are removed

Let's pick [the 10 billion tokens subset of the FineWeb dataset on HuggingFace](https://huggingface.co/datasets/HuggingFaceFW/fineweb). This dataset is used for pretraining general-purpose language models, chatbots, and text generation systems. Examples:

| text                | id      | dump  | url       | date    | file_path | language | language_score | token_count |
| ------------------- | ------- | ----- | --------- | ------- | --------- | -------- | -------------- | ----------- |
| How AP reported ... | <urn..> | CC-.. | http://.. | 2013-.. | s3://..   | en       | 0.972142       | 717         |
| Did you know ...    | <urn..> | CC-.. | http://.. | 2013-.. | s3://..   | en       | 0.947991       | 821         |
| Car Wash For ...    | <urn..> | CC-.. | http://.. | 2013-.. | s3://..   | en       | 0.911518       | 125         |

## 2. Train a tokenizer

Assume that we already clean and preprocess the dataset (eg. remove non-English, deduplicate, filter for quality), let's tokenize it.

The file `src/train_tokenizer.py` loads the dataset and trains a tokenizer using the dataset.

### 2.1. Why do we need to tokenize the dataset?

- Numerical representation: models require input as numbers. Tokenization maps text to token IDs
- Vocabulary building: tokenization defines the set of tokens (vocabulary) the model will understand
- Efficient processing: subword tokenization (eg. BPE) handles rare words and reduces out-of-vocabulary issues. For example, common phrases are usually grouped into a single token, while a rare word will be broken down into several tokens
- Consistent input: tokenization ensures consistent splitting of text, making training and inference reliable

Let's use byte-level Byte-Pair Encoding algorithm to tokenize our dataset.

### 2.2. How does byte-level BPE work?

Ref: https://en.wikipedia.org/wiki/Byte-pair_encoding

### 2.3. Why do we need `batch_iterator` during tokenization process?

`batch_iterator` yields batches of text, allowing the tokenizer to process the data in chunks, because the dataset may be too large to fit into memory at once.

### 2.4. What are some special tokens needed for tokenization process?

| Token    | Description                                                                                                                     |
| -------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `<s>`    | Start-of-sequence token, marks the beginning of a text sequence; helps the model know where input starts                        |
| `</s>`   | End-of-sequence token, marks the end of a text sequence; helps the model know where to stop generating or processing            |
| `<pad>`  | Padding token, used to pad sequences to the same length in a batch; ensures uniform input size for batch processing             |
| `<unk>`  | Unknown token, used for out-of-vocabulary or unrecognized tokens; handles rare or unseen words                                  |
| `<mask>` | Mask token, used for masked language modeling tasks; allows the model to learn context by predicting masked words in a sentence |

### 2.5. How is `<pad>` token used?

Suppose we have a batch of tokenized sequences of different lengths:

- Sequence 1: `<s> Hello world </s>`
- Sequence 2: `<s> How are you? </s>`
- Sequence 3: `<s> Hi </s>`

To process these in a batch, all sequences must have the same length. We pad the shorter ones with `<pad>`:

| Sequence                | Token IDs      |
| ----------------------- | -------------- |
| `<s> Hello world </s>`  | 0 10 20 2 1    |
| `<s> How are you? </s>` | 0 11 12 13 2 1 |
| `<s> Hi </s>`           | 0 14 2 1 1 1   |

Here, `1` is the ID for `<pad>`. So, `<pad>` fills the empty spots so all sequences are the same length for efficient batch processing.

### 2.6. What is `min_frequency` used for?

`min_frequency` sets the minimum number of times a subword or token must appear in the dataset to be included in the vocabulary. It helps filter out rare or noisy tokens, reducing vocabulary size and improving model generalization. Tokens that appear less than `min_frequency` times are replaced with the <unk> (unknown) token.

## 3. Tokenize the dataset

The file `src/tokenize_dataset.py` loads the dataset and tokenizes it using the tokenized trained above.

### 3.1. Why do we need to care about context length when tokenizing the dataset?

The context length (eg. 1024 tokens) is the maximum number of tokens the model can process in a single input sequence. If a tokenized sequence is longer than the context length, we must truncate or split it. If it's shorter, we may need to pad it (with <pad>) for batching.

### 3.2. Why do we need to use batching when tokenizing?

Because we cannot fit the data into memory. In our code, we use a batch size of 1000 rows.

### 3.3. What are methods to tokenize each batch of the dataset?

#### 3.3.1. Method 1: Truncate the batch by context length

```python
def tokenize_function(batch):
  context_length = 1024
  return {
    "input_ids": [
      tokenizer.encode(text)[:context_length] for text in batch["text"]
    ]
  }
```

However, truncation discards tokens beyond the context window, wasting data. Let's try method 2.

#### 3.3.2. Method 2: Split long sequences into multiple chunks of the context length

```python
def tokenize_function(batch):
  max_length = 1024
  input_ids = []
  for text in batch["text"]:
    ids = tokenizer.encode(text)

    # Split into chunks of max_length
    # keep only full-length chunks
    for i in range(0, len(ids), max_length):
      chunk = ids[i:i+max_length]
      if len(chunk) == max_length:
        input_ids.append(chunk)

  return {"input_ids": input_ids}
```

This way, we don't lose any data, and every token in the dataset can be used for training.

However, if a text is longer than 1024 tokens, it is split into multiple 1024-token chunks; any leftover tokens (<1024) are discarded. If a text is shorter than 1024 tokens, it is ignored (since no full chunk can be made). Some data may be wasted, especially for shorter texts. Let's try method 3.

#### 3.3.3. Method 3: Concatenate all texts in the batch before chunking

```python
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

  return {"input_ids": chunks, "labels": chunks.copy()}
```

This method maximizes data usage by concatenating all texts before chunking. Only the very last partial chunk (if any) is discarded. This approach is more efficient for LLM pretraining, especially with many short texts.

Even though chunks may span across multiple texts, separated by EOS tokens, this method is closer to how LLMs are typically pretrained (treating the dataset as a continuous stream).

## 4. Train model

The file `src/compute_params.py` helps to compute the number of params for GPT-2.

The file `src/train_model.py` trains the model.

The file `src/test_pretrained.py` loads the model and completes a prompt.

### 4.1. How many params should our model have?

Our goal is to train a model with < 1B params using GPT-2 architecture. The [Chinchilla/Hoffman scaling laws](https://arxiv.org/abs/2203.15556) suggests that **20 tokens per param** are needed to achieve a notable performance.

If we use the original FineWeb dataset with 10e9 tokens for ~15e6 rows, our model should have 0.5e9 params. However, we are using only 10k rows as the sample dataset, our model should have 0.3e6 params.

### 4.2. How to compute the number of params for GPT-2?

Ref: this answer is generated by GPT-4.1.

To compute the number of parameters for GPT-2 given `n_layer`, `n_head`, `n_embd`, `vocab_size`, and `n_positions`, we need to sum the parameters from:

1. Embedding layers
2. Transformer blocks (per layer)
3. Final layer norm
4. Output (LM head)

#### 4.2.1. Embedding layers

- Token embeddings: `vocab_size * n_embd`
- Position embeddings: `n_positions * n_embd`

#### 4.2.2. Transformer block (per layer)

Each block contains:

- LayerNorm1: `2 * n_embd` (weight + bias)
- Self-Attention:
  - Query, Key, Value: `3 * (n_embd * n_embd + n_embd)` (weights + biases)
  - Output projection: `n_embd * n_embd + n_embd`
- LayerNorm2: `2 * n_embd`
- MLP:
  - First linear: `n_embd * (4 * n_embd) + (4 * n_embd)` (weights + biases)
  - Second linear: `(4 * n_embd) * n_embd + n_embd`

Total per layer:

```
LayerNorms: 2 * (2 * n_embd) = 4 * n_embd
Attention: 3 * (n_embd * n_embd + n_embd) + (n_embd * n_embd + n_embd) = 4 * n_embd * n_embd + 4 * n_embd
MLP: n_embd * 4 * n_embd + 4 * n_embd + 4 * n_embd * n_embd + n_embd = 8 * n_embd * n_embd + 5 * n_embd
Total per layer: 8 * n_embd * n_embd (attn+mlp) + 4 * n_embd * n_embd (attn) + 4 * n_embd (attn) + 5 * n_embd (mlp) + 4 * n_embd (ln) = 12 * n_embd * n_embd + 13 * n_embd
```

But for clarity, sum as:

```
Per layer = 12 * n_embd * n_embd + 13 * n_embd
```

#### 4.2.3. Final LayerNorm

`2 * n_embd`

#### 4.2.4. Output (LM Head)

Usually tied to token embeddings, so no extra parameters. If untied: `vocab_size * n_embd`

#### 4.2.5. Total Parameter Formula

```python
total_params = (
    # Embeddings
    vocab_size * n_embd +
    n_positions * n_embd +
    # Transformer blocks
    n_layer * (12 * n_embd * n_embd + 13 * n_embd) +
    # Final LayerNorm
    2 * n_embd
    # + vocab_size * n_embd  # if output head is untied
)
```

#### 4.2.6. Note

- This formula gives a very close estimate to the official GPT-2 parameter counts
- For most practical purposes, the dominant term is `n_layer * 12 * n_embd^2`
- `n_head` is not used because it affects the model's architecture and parallelism, not the total parameter count, which is governed by `n_embd` and `n_layer`
  - It is a factor in how the attention mechanism is split internally (eg. each head has a dimension of `n_embd // n_head`), but the total number of parameters for the attention layers depends only on `n_embd` (not how it is divided among heads)
  - The weights for query, key, value, and output projections are all of size `[n_embd, n_embd]` regardless of the number of heads.

| Component         | Formula                               |
| ----------------- | ------------------------------------- |
| Token Embedding   | `vocab_size * n_embd`                 |
| Pos Embedding     | `n_positions * n_embd`                |
| Transformer Block | `n_layer * (12*n_embd^2 + 13*n_embd)` |
| Final LayerNorm   | `2 * n_embd`                          |
| Output Head       | (tied, usually not counted)           |

## 5. Instruction tuning

The file `src/prepare_dataset_instruct.py` prepares the dataset for instruction tuning.

The file `src/finetune_instruct.py` starts the instruction tuning process for the pretrained model above.

The file `src/test_instruct.py` tests the pretrained model and the instructed model.

### 5.1. Why do we need instruction tuning?

- Improve usability: models become better at following specific prompts and producing outputs that match what users want
- Generalize to new tasks: models can handle a wider variety of tasks, even those not seen during training, just by giving clear instructions

For example:

```bash
User: Can you tell me who is the current president of USA?
Model: Sure! The current president of USA is ...
```

### 5.2. Why do we need a chat template?

A chat template provides a consistent structure for conversations, making it clear where each user and assistant message begins and ends. This helps the model:

- Recognize roles: distinguish between user and assistant turns
- Learn boundaries: understand where one message ends and another begins
- Generalize better: apply the learned structure to new conversations

For example, using ChatML (Chat Markup Language from OpenAI);

```bash
<|im_start|>user
Can you tell me who is the current president of USA?
<|im_end|>
<|im_start|>assistant
Sure! The current president of USA is ...
<|im_end|>
```

The tokens `<|im_start|>` and `<|im_end|>` are added into the tokenizer vocab and expand the model's token embeddings.
