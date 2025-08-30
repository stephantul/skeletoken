from skeletoken.decase.byte_handlers import text_to_token_str, token_to_bytes


def _determine_collision(
    token: str, is_byte: bool, vocab: set[str], special_tokens: list[str], seen: set[str], remove_collisions: bool
) -> str | None:
    """Determine whether a given token, when lowered, collides with another."""
    if token in special_tokens:
        return token
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

    lowered_token = new_token.casefold()

    if is_byte:
        lowered_token = text_to_token_str(lowered_token)

    # If the token changed but the lowered version was already in vocab, we have a collision.
    if (lowered_token != token) and (lowered_token in vocab or lowered_token in seen):
        return None if remove_collisions else token
    return lowered_token


def decase_vocabulary(
    vocabulary: list[str], special_tokens: list[str], is_byte: bool, remove_collisions: bool = False
) -> list[str]:
    """Lowercase the vocabulary of a tokenizer."""
    uncased_vocab: list[str] = []
    # seen keeps track of lowered tokens that were not in vocab before.
    # e.g., "AB" and "Ab" are in vocab, but they collide after lowercasing
    seen: set[str] = set()
    vocab_set = set(vocabulary)

    for token in vocabulary:
        lowered = _determine_collision(token, is_byte, vocab_set, special_tokens, seen, remove_collisions)
        if lowered is None:
            continue
        uncased_vocab.append(lowered)
        seen.add(lowered)

    return uncased_vocab
