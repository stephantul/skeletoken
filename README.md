
<h2 align="center">
  <img width="35%" alt="A skeleton smoking a cigarette." src="https://raw.githubusercontent.com/stephantul/skeletoken/main/assets/vgogh_skeleton.jpeg"><br/>
</h2>
<h1 align="center"> Skeletoken </h1>

<div align="center">
  <h2>
    <a href="https://pypi.org/project/skeletoken/"><img src="https://img.shields.io/pypi/v/skeletoken?color=f29bdb" alt="Package version"></a>
    <a href="https://codecov.io/gh/stephantul/skeletoken" >
      <img src="https://codecov.io/gh/stephantul/skeletoken/graph/badge.svg?token=DD8BK7OZHG"/>
    </a>
    <a href="https://github.com/stephantul/skeletoken/blob/main/LICENSE">
      <img src="https://img.shields.io/badge/license-MIT-green" alt="License - MIT">
    </a>
    <a href="https://doi.org/10.5281/zenodo.18501953"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.18501953.svg" alt="DOI"></a>
</div>

<div align="center">
  <h2>
    <a href="#installation"><strong>Installation</strong></a> |
    <a href="#example"><strong>Example</strong></a> |
    <a href="#roadmap"><strong>Roadmap</strong></a> |
    <a href="./docs/"><strong>Documentation</strong></a>
  </h2>
</div>

This package contains [`Pydantic`](https://docs.pydantic.dev/latest/) datamodels that fully describe the `tokenizer.json` file used in transformers via [Tokenizers](https://github.com/huggingface/tokenizers). This is useful, because working with this format is complicated.

# Rationale

In one sentence: Validate, edit, and transform Hugging Face tokenizers safely.

The Hugging Face `tokenizers` representation does not reliably allow you to edit tokenizers as a structured object. This means that complex changes to tokenizers require you to edit the `tokenizer.json` file manually. This is annoying, because the format of this file is complicated.

Furthermore, `tokenizers` does not give reasonable errors when parsing a tokenizer fails. It does give line/character numbers, but those point to the _last character_ of the section where the parsing fails. For example, inserting an illegal vocabulary item just tells you that there is an issue in the vocabulary somewhere by pointing out the last character of the vocabulary as the place where the error occurs.

This package contains datamodels (pydantic datamodels) that contain the same constraints as the `tokenizers` package. In other words, if you can create a model in this package, the `tokenizers` package can parse it. This allows you to progressively edit tokenizer json files, all the while getting productive error messages.

# Installation

Install it via pip

```bash
pip install skeletoken
```

# What can it do?

`skeletoken` allows you to:

* validate tokenizer.json with human-readable errors
* edit tokenizers as typed objects (Pydantic)
* apply common transformations (decasing, greedy merges, etc.)
* auto-fix common inconsistencies
* round-trip to `tokenizers` and `transformers`
* apply tokenization changes to `transformers`, `sentence-transformers` and `pylate` models.

# Example

Here's some examples of what skeletoken can do:

## Autofixing a tokenizer

`skeletoken` autofixes any tokenizer you load. See [automatic checks](./docs/2_automatic_checks.md) to see what gets fixed automatically. For example, the [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) tokenizer has a lot of special tokens that are not part of the regular tokenizer vocabulary. This leads to a mismatch between the size of a tokenizer and the number of tokens that tokenizer can produce. `skeletoken` adds these to the vocabulary automatically.

```python
from transformers import AutoTokenizer
from skeletoken import TokenizerModel

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
# Mismatch due to missing special tokens
print(tokenizer.vocab_size)  # 151643
print(len(tokenizer))  # 151669

# Load a model from the hub.
tokenizer_model = TokenizerModel.from_pretrained("Qwen/Qwen3-0.6B")
# Convert the tokenizer to transformers
tokenizer = tokenizer_model.to_transformers()
# All missing special tokens have been added to the vocabulary
print(tokenizer.vocab_size)  # 151669
print(len(tokenizer))  # 151669

```

## Adding components to a tokenizer

`skeletoken` can add components to a tokenizer. First we load one, and inspect it:

```python
from skeletoken import TokenizerModel

# Directly pull a tokenizer from the hub
tokenizer_model = TokenizerModel.from_pretrained("gpt2")

print(tokenizer_model.model.type)
# ModelType.BPE
print(tokenizer_model.pre_tokenizer.type)
# PreTokenizerType.BYTELEVEL
```

We can then add a digit splitter to the tokenizer.

```python
from skeletoken import TokenizerModel
from skeletoken.pretokenizers import DigitsPreTokenizer

model = TokenizerModel.from_pretrained("gpt2")
tok = model.to_tokenizer()

# Create the digits pretokenizer
digits = DigitsPreTokenizer(individual_digits=True)
model = model.add_pre_tokenizer(digits)

new_tok = model.to_tokenizer()
print(tok.encode("hello 123").tokens)
# ['hello', 'Ġ123']
print(new_tok.encode("hello 123").tokens)
# ['hello', 'Ġ', '1', '2', '3']
```

## Decasing a tokenizer

For background, see [this blogpost](https://stephantul.github.io/blog/uncasing/). Decasing is super easy using skeletoken.

```python
from tokenizers import Tokenizer
from skeletoken import TokenizerModel

model_name = "intfloat/multilingual-e5-small"

tokenizer = Tokenizer.from_pretrained(model_name)

print([tokenizer.encode(x).tokens for x in ["Amsterdam", "amsterdam"]])
# [['<s>', '▁Amsterdam', '</s>'], ['<s>', '▁am', 'ster', 'dam', '</s>']]

model = TokenizerModel.from_pretrained(model_name)
model = model.decase_vocabulary()

lower_tokenizer = model.to_tokenizer()
print([lower_tokenizer.encode(x).tokens for x in ["Amsterdam", "amsterdam"]])
# [['<s>', '▁amsterdam', '</s>'], ['<s>', '▁amsterdam', '</s>']]

```

## Making a tokenizer greedy

For background, see [this blog post](https://stephantul.github.io/blog/greedy/). Like decasing, turning any tokenizer into a greedy one is super easy using skeletoken.

```python
from tokenizers import Tokenizer
from skeletoken import TokenizerModel

model_name = "gpt2"

tokenizer = Tokenizer.from_pretrained(model_name)

print([tokenizer.encode(x).tokens for x in [" hellooo", " bluetooth"]])
# [['Ġhell', 'ooo'], ['Ġblu', 'etooth']]

model = TokenizerModel.from_pretrained(model_name)
model = model.make_model_greedy()
greedy_tokenizer = model.to_tokenizer()
print([greedy_tokenizer.encode(x).tokens for x in [" hellooo", " bluetooth"]])
# [['Ġhello', 'oo'], ['Ġblue', 'too', 'th']]

```

# Roadmap

Here's a rough roadmap:

* ✅ Add automated lowercasing (see [blog](https://stephantul.github.io/blog/uncasing/))
* ✅ Add vocabulary changes + checks (e.g., check the merge table if a token is added)
* ✅ Add helper functions for adding modules
* ✅ Add secondary constraints (e.g., if an `AddedToken` refers to a vocabulary item does not exist, we should crash.)
* ✅ Add a front end for the Hugging Face trainer
* ✅ Add automatic model editing
* Consistent tokenizer hashing: instantly know if two tokenizers implement the same thing.
* Add a front end for `sentencepiece` training.

# License

MIT

# Author

Stéphan Tulkens

# Citation

If you use `skeletoken` in your work, please cite:

```bibtex
@software{stephan_tulkens_2026_18501953,
  author       = {Stephan Tulkens},
  title        = {skeletoken},
  month        = feb,
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18501953},
  url          = {https://doi.org/10.5281/zenodo.18501953},
}
```
