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
    subword_prefix: bool,
    word_prefix: bool,
) -> str | None:
    if "�" in decoded:
        return original
    elif added_token := added_token_dict.get(original):
        # If we get here, the token is an added token.
        if added_token.normalized:
            # The added token is already normalized
            return original
        else:
            preprocessed_tokens = preprocessor.preprocess(decoded, word_prefix, subword_prefix)
            if len(preprocessed_tokens) != 1:
                # Check whether the join equals the normalizer-only result. A pure-splitting
                # pretokenizer (e.g. BERT's punctuation splitter) still gives a correct join;
                # a prefix-inserting pretokenizer (e.g. Metaspace) does not.
                joined = "".join(preprocessed_tokens)
                if preprocessor.normalizer is not None:
                    normalized = preprocessor.normalizer.normalize_str(decoded)
                else:
                    normalized = decoded
                if joined != normalized:
                    return original if keep else None
                reprocessed = joined
            else:
                reprocessed = preprocessed_tokens[0]
    else:
        preprocessed_tokens = preprocessor.preprocess(decoded, word_prefix, subword_prefix)
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

    processed_results = [
        _process(
            dt.decoded,
            dt.original,
            added_token_dict,
            new_preprocessor,
            keep,
            dt.had_subword_prefix,
            dt.had_word_prefix,
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
