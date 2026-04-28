from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import Metaspace

from skeletoken.addedtoken import AddedToken
from skeletoken.clean.clean import _process, clean_vocabulary
from skeletoken.preprocessor import Preprocessor


def test_clean() -> None:
    """Test the entire cleaning procedure."""
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


def test_maintain_subwords() -> None:
    """Test whether an subwords are maintained."""
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


def test_remove_multiword() -> None:
    """Test whether an subwords are maintained."""
    p = Preprocessor(word_prefix="_", pretokenizer=Metaspace(replacement="_"))
    vocabulary = ["_dog", "_cat", "_cat hat"]
    decased = clean_vocabulary(
        vocabulary,
        added_tokens=[],
        old_preprocessor=p,
        new_preprocessor=p,
        keep=False,
    )
    assert decased == ["_dog", "_cat", None]


def test_process() -> None:
    """Test the process function with a lot of inputs."""
    p = Preprocessor(subword_prefix="##", word_prefix="P", pretokenizer=Metaspace(replacement="P"))
    x = _process("a", "Z", {}, p, False, False, False)
    assert x == "a"
    x = _process("a", "Z", {}, p, False, True, False)
    assert x == "##a"
    x = _process("a", "Z", {}, p, False, False, True)
    assert x == "Pa"

    # Decoding errors
    x = _process("�", "Z", {}, p, False, False, True)
    assert x == "Z"
    x = _process("�ab", "Z", {}, p, False, False, True)
    assert x == "Z"
    x = _process("�", "Z", {}, p, False, True, False)
    assert x == "Z"
    x = _process("�ab", "Z", {}, p, False, True, True)
    assert x == "Z"

    # Strings that become more than one pretoken are always None if keep = False
    x = _process("a a a", "Z", {}, p, False, True, True)
    assert x == None
    x = _process("a a a", "Z", {}, p, False, False, True)
    assert x == None
    x = _process("a a a", "Z", {}, p, False, True, False)
    assert x == None
    x = _process("a a a", "Z", {}, p, False, False, False)
    assert x == None

    # Strings that become more than one pretoken are always None if keep = False
    x = _process("a a a", "Z", {}, p, True, True, True)
    assert x == "Z"
    x = _process("a a a", "Z", {}, p, True, False, True)
    assert x == "Z"
    x = _process("a a a", "Z", {}, p, True, True, False)
    assert x == "Z"
    x = _process("a a a", "Z", {}, p, True, False, False)
    assert x == "Z"
