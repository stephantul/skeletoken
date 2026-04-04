from typing import TYPE_CHECKING

from skeletoken.addedtoken import AddedToken
from skeletoken.prune.byte_handlers import text_to_token_str, token_to_bytes

if TYPE_CHECKING:
    from skeletoken.preprocessor import Preprocessor


def _determine_collision(
    token: str,
    is_byte: bool,
    vocab: set[str],
    added_tokens: dict[str, AddedToken],
    seen: set[str],
    preprocessor: "Preprocessor",
    keep: bool,
) -> str | None:
    """Determine whether a given token, when processed by a preprocessor, collides with another."""
    if added_token := added_tokens.get(token):
        # If we get here, the token is an added token.
        if added_token.normalized:
            # The added token is already normalized
            return token
        else:
            # This token should be normalized, otherwise it won't be found
            return "".join(preprocessor.preprocess(token))
    if is_byte:
        # Convert token from bytes to a string.
        try:
            token_bytes = token_to_bytes(token)
        except ValueError:
            return token
        try:
            new_token = token_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return token
    else:
        new_token = token

    preprocessed_tokens = preprocessor.preprocess(new_token)

    if len(preprocessed_tokens) > 1:
        return token if keep else None
    preprocessed_token = preprocessed_tokens[0]

    if is_byte:
        preprocessed_token = text_to_token_str(preprocessed_token)

    # If the token changed but the preprocessed version was already in vocab, we have a collision.
    if (preprocessed_token != token) and (preprocessed_token in vocab or preprocessed_token in seen):
        return token if keep else None
    return preprocessed_token


def clean_vocabulary(
    vocabulary: list[str], added_tokens: list[AddedToken], is_byte: bool, preprocessor: "Preprocessor", keep: bool
) -> list[str | None]:
    """Lowercase the vocabulary of a tokenizer."""
    processed_vocab: list[str | None] = []
    # seen keeps track of lowered tokens that were not in vocab before.
    # e.g., "AB" and "Ab" are in vocab, but they collide after lowercasing
    seen: set[str] = set()
    vocab_set = set(vocabulary)

    added_token_dict = {at.content: at for at in added_tokens}

    for token in vocabulary:
        processed = _determine_collision(token, is_byte, vocab_set, added_token_dict, seen, preprocessor, keep)
        processed_vocab.append(processed)
        if isinstance(processed, str):
            seen.add(processed)

    return processed_vocab
