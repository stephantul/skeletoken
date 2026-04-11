from tokenizers.normalizers import Lowercase

from skeletoken.addedtoken import AddedToken
from skeletoken.preprocessor import Preprocessor
from skeletoken.prune.prune import clean_vocabulary


def test_decase() -> None:
    """Test the entire decasing procedure."""
    p = Preprocessor(normalizer=Lowercase())
    vocabulary = ["dog", "CAT", "Cat", "cat", "DOG", "Dog", "SPIN"]
    decased = clean_vocabulary(
        vocabulary,
        added_tokens=[],
        old_preprocessor=p,
        new_preprocessor=p,
        keep=False,
    )
    assert decased == ["dog", None, None, "cat", None, None, "spin"]

    vocabulary = ["dog", "CAT", "Cat", "cat", "DOG", "Dog", "SPIN"]
    decased = clean_vocabulary(
        vocabulary,
        added_tokens=[
            AddedToken(content="SPIN", single_word=True, normalized=True, special=False, lstrip=True, rstrip=True, id=0)
        ],
        old_preprocessor=p,
        new_preprocessor=p,
        keep=False,
    )
    assert decased == ["dog", None, None, "cat", None, None, "SPIN"]


def test_determine_decoder() -> None:
    """Test whether an empty token is removed."""
    p = Preprocessor(subword_prefix="##")
    vocabulary = ["dog", "##cat", "cat"]
    decased = clean_vocabulary(
        vocabulary,
        added_tokens=[],
        old_preprocessor=p,
        new_preprocessor=p,
        keep=False,
    )
    assert decased == ["dog", "##cat", "cat"]
