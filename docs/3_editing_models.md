# Editing models from external libraries

`skeletoken` allows you to edit [`transformers`](https://github.com/huggingface/transformers), [`sentence-transformers`](https://github.com/huggingface/sentence-transformers) and [`pylate`](https://github.com/lightonai/pylate) embedding matrices when the underlying tokenizer is changed.

For example, when decasing a tokenizer, many tokens might disappear or change indices, while the underlying embeddings will not have changed. Similarly, if you remove tokens from a tokenizer, the entire vocabulary can shift, destroying alignment with an embedding matrix, and making your model produce garbage.

The helpers in `skeletoken` automate this process: for each `TokenizerModel`, we keep track of the a `model_delta`. This is an object recording the changes we made to the underlying tokenizer, and what changes to the embedding matrix this will cause. Because we also keep track of the original tokenizer, this allows us to perfectly remap the old embedding matrix to the new one.

Let's first inspect what the `model_delta` looks like:

```python
from skeletoken import TokenizerModel

model_name = "intfloat/multilingual-e5-small"

tokenizer_model = TokenizerModel.from_pretrained(model_name)

print(tokenizer_model.model_delta)

decased = tokenizer_model.decase_vocabulary()
decased.add_token_to_vocabulary("pikachu")

delta = decased.model_delta
# New tokens
print(delta.new_tokens)
# {'pikachu': 229856}
print(delta.token_mapping)
# dict mapping from ints to ints (really long)
print(delta.new_vocabulary_size)
# 229857

```

Here's an example (this requires installing `sentence-transformers`). We will use decasing, as this will allow us to test whether the procedure works, because strings that weren't cased to begin with should have the same vector representation.

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from skeletoken import TokenizerModel
from skeletoken.external.sentence_transformers import reshape_embeddings

model_name = "intfloat/multilingual-e5-small"
model = SentenceTransformer(model_name)
# This tokenizer is cased, so we'll uncase it
tokenizer_model = TokenizerModel.from_pretrained(model_name)
decased = tokenizer_model.decase_vocabulary()

test_string = "this is a test string"

token_ids = model.tokenizer(test_string)['input_ids']
print(token_ids)
# [0, 903, 83, 10, 3034, 79315, 33600, 31, 2]
embeddings = model[0].get_parameter("auto_model.embeddings.word_embeddings.weight")
print(embeddings.shape)
e = embeddings[token_ids].detach().cpu()
# [250037, 384]
x = model.encode(test_string)

decased_model = reshape_embeddings(model, decased)
token_ids = decased_model.tokenizer(test_string)['input_ids']
print(token_ids)
# [0, 826, 82, 10, 2785, 72030, 32122, 2]
embeddings = decased_model[0].get_parameter("auto_model.embeddings.word_embeddings.weight")
print(embeddings.shape)
e2 = embeddings[token_ids].detach().cpu()
# [229856, 384]
x2 = decased_model.encode(test_string)

# Despite the token indices being different, the result vector is the same
assert np.allclose(x, x2)

```

As you can see, this results in the same vector for both models. Note that this does not hold for every lower-case string: if a word had an uppercase form before, it can be that the uppercase token takes precedence.

For example:

```python
from skeletoken import TokenizerModel

model_name = "intfloat/multilingual-e5-small"
model = TokenizerModel.from_pretrained(model_name)
tokenizer = model.to_tokenizer()
print(tokenizer.encode("amsterdam").tokens)
# ['<s>', '▁am', 'ster', 'dam', '</s>']
print(tokenizer.encode("Amsterdam").tokens)
# ['<s>', '▁Amsterdam', '</s>']
model_decased = model.decase_vocabulary()
tokenizer = model_decased.to_tokenizer()
print(tokenizer.encode("amsterdam").tokens)
# ['<s>', '▁amsterdam', '</s>']
print(tokenizer.encode("Amsterdam").tokens)
# ['<s>', '▁amsterdam', '</s>']

```

This is because the token `▁amsterdam` got decased, and hence can be used by the tokenizer when "Amsterdam" is tokenized.
