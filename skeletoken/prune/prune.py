from typing import TYPE_CHECKING

from skeletoken.addedtoken import AddedToken

if TYPE_CHECKING:
    from skeletoken.preprocessor import Preprocessor  # pragma: nocover


def _determine_collision(
    decoded_token: str,
    original_token: str,
    added_tokens: dict[str, AddedToken],
    seen: set[str],
    preprocessor: "Preprocessor",
    keep: bool,
) -> str | None:
    """Determine whether a given token, when processed by a preprocessor, collides with another."""
    # Takes care of tokens like "_" that just encode a space.
    if not decoded_token:
        return decoded_token
    # This is a failed decoding. If it isn't, the original is "�", so that's fine.
    if "�" in decoded_token:
        return original_token
    if added_token := added_tokens.get(original_token):
        # If we get here, the token is an added token.
        if added_token.normalized:
            # The added token is already normalized
            return decoded_token
        token = "".join(preprocessor.preprocess(decoded_token))
        return token

    preprocessed_tokens = preprocessor.preprocess(decoded_token)
    if len(preprocessed_tokens) != 1:
        return decoded_token if keep else None
    preprocessed_token = preprocessed_tokens[0]

    token_changed = preprocessed_token != original_token
    # If the token changed but the preprocessed version was already in vocab, we have a collision.
    if token_changed:
        if not keep or preprocessed_token in seen:
            return decoded_token if keep else None
    return preprocessed_token


def clean_vocabulary(
    vocabulary: list[str],
    added_tokens: list[AddedToken],
    preprocessor: "Preprocessor",
    keep: bool,
) -> list[str | None]:
    """Preprocess the vocabulary of a tokenizer."""
    processed_vocab: list[str | None] = []
    # Decoded tokens. These tokens have no prefix markers.
    decoded = preprocessor.decode_sequences(vocabulary)
    seen = set([x.decoded for x in decoded])
    added_token_dict = {at.content: at for at in added_tokens}

    for decoded_token in decoded:
        processed = _determine_collision(
            decoded_token.decoded, decoded_token.original, added_token_dict, seen, preprocessor, keep
        )
        if isinstance(processed, str):
            seen.add(processed)
        processed_vocab.append(processed)

    return processed_vocab
