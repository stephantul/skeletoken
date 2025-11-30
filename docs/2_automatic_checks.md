# Automatic checks in skeletoken

When loading a tokenizer, `skeletoken` automatically validates and checks your tokenizer, flags weirdnes we can detect, and fixes it when possible.

Here are the checks `skeletoken` performs:

1. For the unk token:
    1. Checks if the `unk` token is in vocabulary, if not, it adds it.
    2. Checks if the `unk` token is marked as the unk token in the tokenizer. Hugging Face tokenizers can mark the `unk` token via `special_tokens_map.json`, which can remove alignment with the actual tokenizer.
    3. When added the `unk` token is added to the `model` of the tokenizer.
2. For the pad token:
    1. Checks if the `pad` token is in vocabulary, if not, it adds it.
    2. Checks if the `pad` token is marked as the unk token in the tokenizer. Hugging Face tokenizers can mark the `pad` token via `special_tokens_map.json`, which can remove alignment with the actual tokenizer.
    3. When added, it is added as a `padding` module with a fixed length padding of 0. If a padding module already exists, the `pad` token in this module is verified or changed to the hugging face one.
    4. If you already define a padding token, we check whether it actually occurs in the vocabulary, and whether the current index you define actually points to that padding token.
3. For all added tokens:
    1. Check whether the added token occurs in the vocabulary at the correct index. If it does not, this can lead to collisions. These are also auto-fixed. If the item occurs at a different index, we change it to that index. If the item does not occur at all, we add it to the vocabulary at a new index.

All of these emitted messages are `warning`s and can be silenced by correctly setting your log level.
