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

The file `src/prepare_dataset.py` downloads the dataset and creates a sample dataset for testing the code faster.

### Why do we need a dataset?

The dataset provides examples for the model to learn language patterns such as:

- Learn grammar: the model observes correct English grammar in the dataset and learns to reproduce it
- Build vocabulary: the model observes a wide range of words and phrases
- Understand context: the model observes how sentences and ideas flow and learns to generate coherent responses

### What are criteria to consider a dataset "good enough"?

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

### Why do we need to tokenize the dataset?

- Numerical representation: models require input as numbers. Tokenization maps text to token IDs
- Vocabulary building: tokenization defines the set of tokens (vocabulary) the model will understand
- Efficient processing: subword tokenization (eg. BPE) handles rare words and reduces out-of-vocabulary issues. For example, common phrases are usually grouped into a single token, while a rare word will be broken down into several tokens
- Consistent input: tokenization ensures consistent splitting of text, making training and inference reliable

Let's use byte-level Byte-Pair Encoding algorithm to tokenize our dataset.

### How does byte-level BPE work?

Ref: https://en.wikipedia.org/wiki/Byte-pair_encoding

### Why do we need `batch_iterator` during tokenization process?

`batch_iterator` yields batches of text, allowing the tokenizer to process the data in chunks, because the dataset may be too large to fit into memory at once.

### What are some special tokens needed for tokenization process?

| Token    | Description                                                                                                                     |
| -------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `<s>`    | Start-of-sequence token, marks the beginning of a text sequence; helps the model know where input starts                        |
| `</s>`   | End-of-sequence token, marks the end of a text sequence; helps the model know where to stop generating or processing            |
| `<pad>`  | Padding token, used to pad sequences to the same length in a batch; ensures uniform input size for batch processing             |
| `<unk>`  | Unknown token, used for out-of-vocabulary or unrecognized tokens; handles rare or unseen words                                  |
| `<mask>` | Mask token, used for masked language modeling tasks; allows the model to learn context by predicting masked words in a sentence |

### How is `<pad>` token used?

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

### What is `min_frequency` used for?

`min_frequency` sets the minimum number of times a subword or token must appear in the dataset to be included in the vocabulary. It helps filter out rare or noisy tokens, reducing vocabulary size and improving model generalization. Tokens that appear less than `min_frequency` times are replaced with the <unk> (unknown) token.

## 3. Tokenize the dataset

The file `src/tokenize_dataset.py` loads the dataset and tokenizes it using the tokenized trained above.

### Why do we need to care about context length when tokenizing the dataset?

The context length (eg. 1024 tokens) is the maximum number of tokens the model can process in a single input sequence. If a tokenized sequence is longer than the context length, we must truncate or split it. If it’s shorter, we may need to pad it (with <pad>) for batching.

### Why do we need to use batching when tokenizing?

Because we cannot fit the data into memory. In our code, we use a batch size of 1000 rows.

### What are methods to tokenize each batch of the dataset?

#### Method 1: Truncate the batch by context length

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

#### Method 2: Split long sequences into multiple chunks of the context length

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

This way, we don’t lose any data, and every token in the dataset can be used for training.

However, if a text is longer than 1024 tokens, it is split into multiple 1024-token chunks; any leftover tokens (<1024) are discarded. If a text is shorter than 1024 tokens, it is ignored (since no full chunk can be made). Some data may be wasted, especially for shorter texts. Let's try method 3.

#### Method 3: Concatenate all texts in the batch before chunking

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
