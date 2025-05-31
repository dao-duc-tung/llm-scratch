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

## Prepare a dataset

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

## Tokenize the dataset

Assume that we already clean and preprocess the dataset (eg. remove non-English, deduplicate, filter for quality), let's tokenize it.

The file `src/tokenize_dataset.py` loads the dataset, tokenizes it and saves the tokenized dataset to disk.

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
