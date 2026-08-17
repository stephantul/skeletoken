from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import BertPreTokenizer as HFBertPreTokenizer
from tokenizers.pre_tokenizers import Metaspace, WhitespaceSplit

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
    p = Preprocessor(continuing_subword_prefix="##")
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
    p = Preprocessor(initial_subword_prefix="_", pretokenizer=Metaspace(replacement="_"))
    vocabulary = ["_dog", "_cat", "_cat hat"]
    decased = clean_vocabulary(
        vocabulary,
        added_tokens=[],
        old_preprocessor=p,
        new_preprocessor=p,
        keep=False,
    )
    assert decased == ["_dog", "_cat", None]


def test_infer_word_initial_when_introducing_initial_subword_prefix() -> None:
    """When a word prefix is newly introduced, bare (non-subword-prefixed) tokens are word-initial."""
    old = Preprocessor(continuing_subword_prefix="##")
    new = Preprocessor(
        continuing_subword_prefix="##", initial_subword_prefix="▁", pretokenizer=Metaspace(replacement="▁", split=False)
    )
    vocabulary = ["dog", "##cat", "cat"]
    result = clean_vocabulary(
        vocabulary,
        added_tokens=[],
        old_preprocessor=old,
        new_preprocessor=new,
        keep=False,
    )
    assert result == ["▁dog", "##cat", "▁cat"]


def test_no_infer_word_initial_without_continuing_subword_prefix_scheme() -> None:
    """Without an old subword prefix scheme, there's no signal to infer word-initial-ness from."""
    old = Preprocessor()
    new = Preprocessor(initial_subword_prefix="▁", pretokenizer=Metaspace(replacement="▁", split=False))
    vocabulary = ["dog", "cat"]
    result = clean_vocabulary(
        vocabulary,
        added_tokens=[],
        old_preprocessor=old,
        new_preprocessor=new,
        keep=False,
    )
    assert result == ["dog", "cat"]


def test_process() -> None:
    """Test the process function with a lot of inputs."""
    p = Preprocessor(
        continuing_subword_prefix="##", initial_subword_prefix="P", pretokenizer=Metaspace(replacement="P")
    )
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


def test_process_added_token_pure_split_join() -> None:
    """Test _process for an added token (normalized=False) whose multi-token split rejoins cleanly.

    BertPreTokenizer splits "a.b" into ["a", ".", "b"] without inserting any extra characters,
    so joined ("a.b") equals decoded ("a.b") and the reprocessed = joined branch is reached.
    """
    p = Preprocessor(pretokenizer=HFBertPreTokenizer())
    at = AddedToken(content="a.b", single_word=False, normalized=False, special=False, lstrip=False, rstrip=False, id=0)
    added_token_dict = {"a.b": at}
    result = _process("a.b", "a.b", added_token_dict, p, False, False, False)
    assert result == "a.b"


def test_process_added_token_returns_original_regardless_of_split() -> None:
    """Added tokens are matched against input before the normalizer/pretokenizer ever runs."""
    p = Preprocessor(
        normalizer=Lowercase(),
        initial_subword_prefix="▁",
        pretokenizer=Metaspace(replacement="▁"),
    )
    at = AddedToken(
        content="▁HELLO WORLD", single_word=False, normalized=False, special=False, lstrip=False, rstrip=False, id=0
    )
    added_token_dict = {"▁HELLO WORLD": at}
    assert _process("HELLO WORLD", "▁HELLO WORLD", added_token_dict, p, False, False, True) == "▁HELLO WORLD"
    assert _process("HELLO WORLD", "▁HELLO WORLD", added_token_dict, p, True, False, True) == "▁HELLO WORLD"


def test_process_added_token_empty_preprocess() -> None:
    """Added token whose decoded form yields no pretokens still returns unchanged."""
    p = Preprocessor(pretokenizer=WhitespaceSplit())
    at = AddedToken(content="[WS]", single_word=True, normalized=False, special=True, lstrip=False, rstrip=False, id=0)
    added_token_dict = {"[WS]": at}

    x = _process(" ", "[WS]", added_token_dict, p, False, False, False)
    assert x == "[WS]"

    x = _process(" ", "[WS]", added_token_dict, p, True, False, False)
    assert x == "[WS]"
