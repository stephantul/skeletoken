# Tokenizer datamodels

This package contains [`Pydantic`](https://docs.pydantic.dev/latest/) datamodels that fully describe the `tokenizer.json` file used in transformers via [Tokenizers](https://github.com/huggingface/tokenizers). This is useful, because working with this format is complicated.

# Rationale

The Hugging Face `tokenizers` representation does not reliably allow you to edit tokenizers as a structured object. This means that complex changes to tokenizers require you to edit the `tokenizer.json` file manually. This is annoying, because the format of this file is complicated.

Furthermore, `tokenizers` does not give reasonable errors when parsing a tokenizer fails. It does give line/character numbers, but those point to the _last character_ of the section where the parsing fails. For example, inserting an illegal vocabulary item just tells you that there is an issue in the vocabulary somewhere by pointing out the last character of the vocabulary as the place where the error occurs.

This package contains datamodels (pydantic Basemodels) that contain the same constraints as the `tokenizers` package. In other words, if you can create a model in this package, the `tokenizers` package can parse it. This allows you to progressively edit tokenizer json files, all the while getting productive error messages.

# Example

```python
from tokenizers import Tokenizer
from tokenizerdatamodels import TokenizerModel

tok = Tokenizer.from_pretrained("gpt2")
tokenizer_model = TokenizerModel.model_validate_json(tok.to_str())

print(tokenizer_model.model.type)
# ModelType.BPE
print(tokenizer_model.pre_tokenizer.type)
# PreTokenizerType.BYTELEVEL
```

Ok, now let's add a digit splitter to the tokenizer.

```python
from tokenizers import Tokenizer
from tokenizerdatamodels import TokenizerModel
from tokenizerdatamodels.pretokenizers import DigitsPreTokenizer, PretokenizerSequence

tok = Tokenizer.from_pretrained("gpt2")
tokenizer_model = TokenizerModel.model_validate_json(tok.to_str())

# The gpt tokenizer only has a single pretokenizer.
# So we need to add a sequence
digits = DigitsPreTokenizer(individual_digits=True)
sequence = PretokenizerSequence(pretokenizers=[tokenizer_model.pre_tokenizer, digits])
tokenizer_model.pre_tokenizer = sequence

new_tok = Tokenizer.from_str(tokenizer_model.model_dump_json())
print(tok.encode("hello 123").tokens)
# ['hello', 'Ġ123']
print(new_tok.encode("hello 123").tokens)
# ['hello', 'Ġ', '1', '2', '3']
```

The example above is still pretty rough. In the future, you'll be able to add pretokenizers with something like `.add_pretokenizer`, which then would automatically add a sequence if necessary.

# Roadmap

Here's a rough roadmap:

* Add automated lowercasing
* Add vocabulary changes + checks (e.g., check the merge table if a token is added)
* Add helper functions (e.g., the aforementioned `.add_pretokenizer`)
* Add secondary constraints (e.g., if an `AddedToken` refers to a vocabulary item does not exist, we should crash.)

# Installation

I only offer git right now:

```bash
pip install git+https://github.com/stephantul/tokenizerdatamodels.git
```


# License

MIT

# Author

Stéphan Tulkens
