from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skeletoken.base import TokenizerModel  # pragma: no cover


def empty_tokenizer() -> "TokenizerModel":
    """Create an empty tokenizer model. Used to start from scratch."""
    from skeletoken.base import TokenizerModel
    from skeletoken.merges import Merges
    from skeletoken.models import BPE
    from skeletoken.vocabulary import Vocabulary

    return TokenizerModel(
        model=BPE(
            merges=Merges([]),
            vocab=Vocabulary(root={}),
            dropout=None,
            unk_token=None,
            continuing_subword_prefix=None,
            end_of_word_suffix=None,
            fuse_unk=False,
            byte_fallback=False,
            ignore_merges=False,
        ),
        normalizer=None,
        pre_tokenizer=None,
        post_processor=None,
        decoder=None,
    )
