from collections.abc import Iterator

from tokenizers import AddedToken as TokenizersAddedToken
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordPieceTrainer

from skeletoken.addedtoken import AddedToken
from skeletoken.base import TokenizerModel
from skeletoken.models import BPE, ModelType, Unigram, WordPiece


def _convert_to_added_tokens(tokens: list[AddedToken]) -> list[TokenizersAddedToken]:
    """Converts a list of strings to a list of AddedToken objects."""
    return [
        TokenizersAddedToken(
            content=token.content, single_word=token.single_word, lstrip=token.lstrip, rstrip=token.rstrip
        )
        for token in tokens
    ]


def train_tokenizer(model: TokenizerModel, data: Iterator[str], vocab_size: int) -> TokenizerModel:
    """
    Trains the tokenizer model on the provided data.

    Parameters
    ----------
    model : TokenizerModel
        The tokenizer model to be trained.
    data : Iterator[str]
        The training data as an iterator of strings.
    vocab_size : int
        The desired vocabulary size.

    Returns
    -------
    TokenizerModel
        The trained tokenizer model.

    Raises
    ------
    ValueError
        If the model type is unknown.

    """
    model_type = model.model.type
    tokenizer = model.to_tokenizer()

    special_tokens = _convert_to_added_tokens(model.added_tokens.get_special_tokens())

    if model_type == ModelType.BPE:
        assert isinstance(model.model, BPE)
        trainer = BpeTrainer(
            vocab_size=vocab_size,  # type: ignore[arg-type]
            special_tokens=special_tokens,  # type: ignore[arg-type]
            continuing_subword_prefix=model.model.continuing_subword_prefix,  # type: ignore[arg-type]
            end_of_word_suffix=model.model.end_of_word_suffix,  # type: ignore[arg-type]
        )
    elif model_type == ModelType.UNIGRAM:
        assert isinstance(model.model, Unigram)
        trainer = UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            unk_token=model.model.unk_token,
        )
    elif model_type == ModelType.WORDPIECE:
        assert isinstance(model.model, WordPiece)
        trainer = WordPieceTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            continuing_subword_prefix=model.model.continuing_subword_prefix,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    tokenizer.train_from_iterator(data, trainer)
    return TokenizerModel.from_tokenizer(tokenizer)
