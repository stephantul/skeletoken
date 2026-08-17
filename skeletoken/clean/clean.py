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
    continuing_subword_prefix: bool,
    initial_subword_prefix: bool,
) -> str | None:
    if "�" in decoded:
        return original
    elif original in added_token_dict:
        return original
    else:
        preprocessed_tokens = preprocessor.preprocess(
            decoded, initial_subword_prefix, continuing_subword_prefix, empty_sequence_is_token=True
        )
        if len(preprocessed_tokens) != 1:
            return original if keep else None
        reprocessed = preprocessed_tokens[0]

    return reprocessed


def clean_vocabulary(
    vocabulary: list[str],
    added_tokens: list[AddedToken],
    old_preprocessor: "Preprocessor",
    new_preprocessor: "Preprocessor",
    keep: bool,
) -> list[str | None]:
    """Preprocess the vocabulary of a tokenizer."""
    # Decoded tokens. These tokens have no prefix markers.
    decoded_sequences = old_preprocessor.decode_sequences(vocabulary)
    added_token_dict = {at.content: at for at in added_tokens}

    # An initial subword prefix is being introduced where none existed before. If the model already
    # distinguishes continuation pieces via a continuing subword prefix (e.g. WordPiece's "##"), the
    # absence of that marker reliably means the token was word-initial, so it should now receive the
    # new initial subword prefix too.
    can_infer_word_initial = (
        old_preprocessor.initial_subword_prefix is None
        and new_preprocessor.initial_subword_prefix is not None
        and old_preprocessor.continuing_subword_prefix is not None
    )

    processed_results = [
        _process(
            dt.decoded,
            dt.original,
            added_token_dict,
            new_preprocessor,
            keep,
            dt.had_continuing_subword_prefix,
            not dt.had_continuing_subword_prefix if can_infer_word_initial else dt.had_initial_subword_prefix,
        )
        for dt in decoded_sequences
    ]

    # Unchanged tokens (processed == original) get deduplication priority.
    seen: dict[str, int] = {
        r: i
        for i, (r, dt) in enumerate(zip(processed_results, decoded_sequences, strict=False))
        if r is not None and r == dt.original
    }

    processed_vocab: list[str | None] = []
    for i, (processed, dt) in enumerate(zip(processed_results, decoded_sequences, strict=False)):
        if processed is None:
            processed_vocab.append(None)
            continue
        index = seen.get(processed)
        if index is not None and index != i:
            processed_vocab.append(dt.original if keep else None)
            continue
        seen[processed] = i
        processed_vocab.append(processed)

    return processed_vocab
