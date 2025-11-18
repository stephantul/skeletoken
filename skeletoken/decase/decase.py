from skeletoken.addedtoken import AddedToken
from skeletoken.decase.byte_handlers import text_to_token_str, token_to_bytes


def _determine_collision(
    token: str, is_byte: bool, vocab: set[str], added_tokens: dict[str, AddedToken], seen: set[str], lower: bool
) -> str | None:
    """Determine whether a given token, when lowered, collides with another."""
    if added_token := added_tokens.get(token):
        # If we get here, the token is an added token.
        if added_token.normalized:
            # The added token is already normalized
            return token
        else:
            # This token will be normalized
            return token.lower()
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

    lowered_token = new_token.lower()

    if not lower and lowered_token != new_token:
        return None

    if is_byte:
        lowered_token = text_to_token_str(lowered_token)

    # If the token changed but the lowered version was already in vocab, we have a collision.
    if (lowered_token != token) and (lowered_token in vocab or lowered_token in seen):
        return None
    return lowered_token


def decase_vocabulary(
    vocabulary: list[str], added_tokens: list[AddedToken], is_byte: bool, lower: bool
) -> list[str | None]:
    """Lowercase the vocabulary of a tokenizer."""
    uncased_vocab: list[str | None] = []
    # seen keeps track of lowered tokens that were not in vocab before.
    # e.g., "AB" and "Ab" are in vocab, but they collide after lowercasing
    seen: set[str] = set()
    vocab_set = set(vocabulary)

    added_token_dict = {at.content: at for at in added_tokens}

    for token in vocabulary:
        lowered = _determine_collision(token, is_byte, vocab_set, added_token_dict, seen, lower)
        uncased_vocab.append(lowered)
        if isinstance(lowered, str):
            seen.add(lowered)

    return uncased_vocab
