# Saving and loading a `TokenizerModel`

`skeletoken` supports saving and loading from a variety of formats, including.

1. transformers.PreTrainedTokenizerFast
2. Tokenizers

In all cases, loading from remote and local directories is handled by `skeletoken`, so no manual loading is necessary.

## Loading a `TokenizerModel`

Loading a tokenizer can be done using the `from_pretrained` function.

```python
from skeletoken import TokenizerModel
from transformers import AutoTokenizer

# Loads the mixedbread/mxbai/embed-large-v1 tokenizer
model = TokenizerModel.from_pretrained("mixedbread-ai/mxbai-embed-large-v1")

# Loading a local tokenizer
tokenizer = AutoTokenizer.from_pretrained("mixedbread-ai/mxbai-embed-large-v1")
tokenizer.save_pretrained("my_temporary_dir")

model = TokenizerModel.from_pretrained("my_temporary_dir")
# Explicit for hf tokenizers
model = TokenizerModel.from_transformers("my_temporary_dir")

# Works for tokenizer.json as well
model = TokenizerModel.from_pretrained("my_temporary_dir/tokenizer.json")

```

When in doubt, always use the folder, i.e., load from `my_temporary_dir` instead of `my_temporary_dir/tokenizer.json`. If present, this will give `skeletoken` extra context and attributes to work with. If there is no information present, it will be equivalent to `my_temporary_dir/tokenizer.json`.

You can also load a `TokenizerModel` from an in-memory tokenizer. This is otherwise equivalent to `from_pretrained`, above, but can be useful if you load your model in another way.

```python
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from skeletoken import TokenizerModel

hf_tokenizer = AutoTokenizer.from_pretrained("mixedbread-ai/mxbai-embed-large-v1")

model = TokenizerModel.from_transformers_tokenizer(hf_tokenizer)

tokenizer = Tokenizer.from_pretrained("mixedbread-ai/mxbai-embed-large-v1")
model = TokenizerModel.from_tokenizer(tokenizer)

```

## Converting a `TokenizerModel` to a tokenizer

There's a couple of ways of turning a `TokenizerModel` into a tokenizer:

```python
from skeletoken import TokenizerModel
from transformers import AutoTokenizer

# Loads the mixedbread/mxbai/embed-large-v1 tokenizer
model = TokenizerModel.from_pretrained("mixedbread-ai/mxbai-embed-large-v1")

bare_tokenizer = model.to_tokenizer()
transformers_tokenizer = model.to_transformers()

```

For the transformers tokenizer conversion, is important to keep alignment with the tokenizer class. This is handled automatically, but just in case it isn't, you can pass the original tokenizer class to the converter.

```python
from transformers import AutoTokenizer

from skeletoken import TokenizerModel

tokenizer = AutoTokenizer.from_pretrained("gpt2")
print(type(tokenizer))
# Will show
# <class 'transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast'>
model = TokenizerModel.from_transformers_tokenizer(tokenizer)

other_tokenizer = model.to_transformers(tokenizer_class=type(tokenizer))
print(type(other_tokenizer))

```

## Saving a `TokenizerModel`

There's many ways to save a tokenizer.

### As a JSON object or string

While it is possible to directly dump the tokenizer object using pydantic's built-in model dump, it is actually preferred to first turn it into a tokenizer. That way, you get instant validation for your tokenizer. So, while this is possible:

```python
from tokenizers import Tokenizer

from skeletoken import TokenizerModel

model = TokenizerModel.from_pretrained("gpt2")
string = model.model_dump_json()

tokenizer = Tokenizer.from_str(string)

with open("temp.json", "w") as f:
    f.write(string)

tokenizer = Tokenizer.from_file("temp.json")

```

The preferred way to do it is by first converting, and then saving.

```python
from tokenizers import Tokenizer

from skeletoken import TokenizerModel

model = TokenizerModel.from_pretrained("gpt2")
tokenizer = model.to_tokenizer()

tokenizer.save("hello.json")

# Or to a HF tokenizer
tokenizer = model.to_transformers()

tokenizer.save_pretrained("my_tokenizer")

```
