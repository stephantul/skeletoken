from typing import TYPE_CHECKING

from skeletoken.addedtoken import AddedToken

if TYPE_CHECKING:
    from skeletoken.preprocessor import Preprocessor  # pragma: nocover


def _process(
    decoded: str,
    original: str,
    added_token_dict: dict[str, AddedToken],
    preprocessor: "Preprocessor",
    keep: bool,
    subword_prefix: str | None,
    word_prefix: str | None,
) -> str | None:
    if "�" in decoded:
        return original
    elif added_token := added_token_dict.get(original):
        # If we get here, the token is an added token.
        if added_token.normalized:
            # The added token is already normalized
            return decoded
        else:
            reprocessed = "".join(preprocessor.preprocess(decoded))
    else:
        preprocessed_tokens = preprocessor.preprocess(decoded)
        if len(preprocessed_tokens) != 1:
            return original if keep else None
        reprocessed = preprocessed_tokens[0]
    if subword_prefix is not None:
        reprocessed = f"{preprocessor.subword_prefix}{reprocessed}"
    if word_prefix is None and preprocessor.word_prefix:
        reprocessed = reprocessed.removeprefix(preprocessor.word_prefix)

    return reprocessed


def clean_vocabulary(
    vocabulary: list[str],
    added_tokens: list[AddedToken],
    old_preprocessor: "Preprocessor",
    new_preprocessor: "Preprocessor",
    keep: bool,
) -> list[str | None]:
    """Preprocess the vocabulary of a tokenizer."""
    processed_vocab: list[str | None] = []
    # Decoded tokens. These tokens have no prefix markers.
    decoded_sequences = old_preprocessor.decode_sequences(vocabulary)
    added_token_dict = {at.content: at for at in added_tokens}

    new_vocab: list[tuple[str | None, str]] = []

    for decoded_token in decoded_sequences:
        decoded = decoded_token.decoded
        original = decoded_token.original
        reprocessed = _process(
            decoded,
            original,
            added_token_dict,
            new_preprocessor,
            keep,
            decoded_token.subword_prefix,
            decoded_token.word_prefix,
        )
        new_vocab.append((reprocessed, original))

    # We give preference for unchanged tokens. These should not be removed.
    seen: dict[str, int] = {x: i for i, (x, y) in enumerate(new_vocab) if x is not None and x == y}

    processed_vocab = []
    for i, (processed, original) in enumerate(new_vocab):
        if processed is None:
            processed_vocab.append(processed)
            continue
        index = seen.get(processed)
        if index is not None and index != i:
            processed_vocab.append(original if keep else None)
            continue
        seen[processed] = i
        processed_vocab.append(processed)

    return processed_vocab
