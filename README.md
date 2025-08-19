
<h2 align="center">
  <img width="35%" alt="Skeleton by van Gogh. Source: https://nl.wikipedia.org/wiki/Kop_van_een_skelet_met_brandende_sigaret#/media/Bestand:Vincent_van_Gogh_-_Head_of_a_skeleton_with_a_burning_cigarette_-_Google_Art_Project.jpg" src="assets/vgogh_skeleton.jpeg"><br/>
</h2>
<h1 align="center"> Skeletoken </h2>

This package contains [`Pydantic`](https://docs.pydantic.dev/latest/) datamodels that fully describe the `tokenizer.json` file used in transformers via [Tokenizers](https://github.com/huggingface/tokenizers). This is useful, because working with this format is complicated.

# Rationale

The Hugging Face `tokenizers` representation does not reliably allow you to edit tokenizers as a structured object. This means that complex changes to tokenizers require you to edit the `tokenizer.json` file manually. This is annoying, because the format of this file is complicated.

Furthermore, `tokenizers` does not give reasonable errors when parsing a tokenizer fails. It does give line/character numbers, but those point to the _last character_ of the section where the parsing fails. For example, inserting an illegal vocabulary item just tells you that there is an issue in the vocabulary somewhere by pointing out the last character of the vocabulary as the place where the error occurs.

This package contains datamodels (pydantic Basemodels) that contain the same constraints as the `tokenizers` package. In other words, if you can create a model in this package, the `tokenizers` package can parse it. This allows you to progressively edit tokenizer json files, all the while getting productive error messages.

# Example

```python
from tokenizers import Tokenizer
from skeletoken import TokenizerModel

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
from skeletoken import TokenizerModel
from skeletoken.pretokenizers import DigitsPreTokenizer

tok = Tokenizer.from_pretrained("gpt2")
tokenizer_model = TokenizerModel.from_pretrained("gpt2")

# Create the digits pretokenizer
digits = DigitsPreTokenizer(individual_digits=True)
tokenizer_model.add_pre_tokenizer(digits)

new_tok = tokenizer_model.to_tokenizer()
print(tok.encode("hello 123").tokens)
# ['hello', 'Ġ123']
print(new_tok.encode("hello 123").tokens)
# ['hello', 'Ġ', '1', '2', '3']
```

# Roadmap

Here's a rough roadmap:

* Add automated lowercasing (see [blog](https://stephantul.github.io/tokenization/casing/2025/08/01/uncasing/))
* Add vocabulary changes + checks (e.g., check the merge table if a token is added)
* ✅ Add helper functions for adding modules
* Add secondary constraints (e.g., if an `AddedToken` refers to a vocabulary item does not exist, we should crash.)

# Installation

I only offer git right now:

```bash
pip install git+https://github.com/stephantul/skeletoken.git
```

# License

MIT

# Author

Stéphan Tulkens
