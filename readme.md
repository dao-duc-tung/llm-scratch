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

Code files:

- `prepare_dataset.py`: download the dataset and create a sample dataset for testing the code faster

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
