from tokenizers import Regex
from tokenizers.normalizers import Lowercase, Replace
from tokenizers.pre_tokenizers import Digits

from skeletoken.addedtoken import AddedToken
from skeletoken.preprocessor import Preprocessor
from skeletoken.prune.prune import _determine_collision, clean_vocabulary


def test_determine_collision() -> None:
    """Test the collision detection in the vocabulary."""
    vocabulary = {"dog", "CAT", "Cat", "cat"}
    # Unchanged (already in vocab)
    p = Preprocessor(normalizer=Lowercase())
    assert (
        _determine_collision(
            decoded_token="dog",
            original_token="dog",
            added_tokens={},
            seen=vocabulary,
            preprocessor=p,
            keep=True,
        )
        == "dog"
    )
    # Unchanged (already in vocab)
    assert (
        _determine_collision(
            decoded_token="DOG",
            original_token="DOG",
            added_tokens={},
            seen=vocabulary,
            preprocessor=p,
            keep=False,
        )
        == None
    )
    # Unchanged (already lowered)
    assert (
        _determine_collision(
            decoded_token="cat",
            original_token="cat",
            added_tokens={},
            seen=vocabulary,
            preprocessor=p,
            keep=False,
        )
        == "cat"
    )
    # Unchanged (already in vocab)
    assert (
        _determine_collision(
            decoded_token="CAT",
            original_token="CAT",
            added_tokens={},
            seen=vocabulary,
            preprocessor=p,
            keep=False,
        )
        == None
    )
    # Unchanged (already in vocab)
    assert (
        _determine_collision(
            decoded_token="cAT",
            original_token="cAT",
            added_tokens={},
            seen=vocabulary,
            preprocessor=p,
            keep=False,
        )
        == None
    )
    # Additional test (new token)
    assert (
        _determine_collision(
            decoded_token="SPIN",
            original_token="SPIN",
            added_tokens={},
            seen=vocabulary,
            preprocessor=p,
            keep=False,
        )
        == None
    )
    # Unchanged (already in seen)
    assert (
        _determine_collision(
            decoded_token="SPIN",
            original_token="SPIN",
            added_tokens={},
            seen={"spin"} | vocabulary,
            preprocessor=p,
            keep=False,
        )
        == None
    )
    # Unchanged (special token)
    assert (
        _determine_collision(
            decoded_token="Spin",
            original_token="Spin",
            added_tokens={
                "Spin": AddedToken(
                    content="Spin", single_word=True, normalized=True, special=False, lstrip=True, rstrip=True, id=0
                )
            },
            seen=vocabulary,
            preprocessor=p,
            keep=False,
        )
        == "Spin"
    )

    # Byte stuff
    assert (
        _determine_collision(
            decoded_token="dog",
            original_token="dog",
            added_tokens={},
            seen=vocabulary,
            preprocessor=p,
            keep=False,
        )
        == "dog"
    )
    assert (
        _determine_collision(
            decoded_token="DOG",
            original_token="DOG",
            added_tokens={},
            seen=vocabulary,
            preprocessor=p,
            keep=False,
        )
        == None
    )
    assert (
        _determine_collision(
            decoded_token="cat",
            original_token="cat",
            added_tokens={},
            seen=vocabulary,
            preprocessor=p,
            keep=False,
        )
        == "cat"
    )
    assert (
        _determine_collision(
            decoded_token="CAT",
            original_token="CAT",
            added_tokens={},
            seen=vocabulary,
            preprocessor=p,
            keep=False,
        )
        == None
    )
    assert (
        _determine_collision(
            decoded_token="cAT",
            original_token="cAT",
            added_tokens={},
            seen=vocabulary,
            preprocessor=p,
            keep=False,
        )
        == None
    )
    assert (
        _determine_collision(
            decoded_token="SPIN",
            original_token="SPIN",
            added_tokens={},
            seen=vocabulary,
            preprocessor=p,
            keep=False,
        )
        == None
    )
    assert (
        _determine_collision(
            decoded_token="SPIN",
            original_token="SPIN",
            added_tokens={},
            seen={"spin"} | vocabulary,
            preprocessor=p,
            keep=False,
        )
        == None
    )
    assert (
        _determine_collision(
            decoded_token="Spin",
            original_token="Spin",
            added_tokens={
                "Spin": AddedToken(
                    content="Spin", single_word=True, normalized=True, special=False, lstrip=True, rstrip=True, id=0
                )
            },
            seen=vocabulary,
            preprocessor=p,
            keep=False,
        )
        == "Spin"
    )

    # Unicode error
    assert (
        _determine_collision(
            original_token="\x80",
            decoded_token="\x80",
            added_tokens={},
            seen=vocabulary,
            preprocessor=p,
            keep=False,
        )
        == "\x80"
    )

    # No unicode error
    assert (
        _determine_collision(
            original_token="\x80",
            decoded_token="\x80",
            added_tokens={},
            seen=vocabulary,
            preprocessor=p,
            keep=False,
        )
        == "\x80"
    )

    # Decode error
    assert (
        _determine_collision(
            original_token="\xa1",
            decoded_token="\xa1",
            added_tokens={},
            seen=vocabulary,
            preprocessor=p,
            keep=False,
        )
        == "\xa1"
    )


def test_determine_collision_lower() -> None:
    """Test the collision detection in the vocabulary."""
    p = Preprocessor(normalizer=Lowercase())
    vocabulary = {"dog", "CAT", "Cat", "cat"}
    # Unchanged (already in vocab)
    assert (
        _determine_collision(
            decoded_token="dog",
            original_token="dog",
            added_tokens={},
            seen=vocabulary,
            preprocessor=p,
            keep=True,
        )
        == "dog"
    )
    # Unchanged (already in vocab)
    assert (
        _determine_collision(
            decoded_token="DOG",
            original_token="DOG",
            added_tokens={},
            seen=vocabulary,
            preprocessor=p,
            keep=False,
        )
        == None
    )
    # Unchanged (already lowered)
    assert (
        _determine_collision(
            decoded_token="cat",
            original_token="cat",
            added_tokens={},
            seen=vocabulary,
            preprocessor=p,
            keep=True,
        )
        == "cat"
    )
    # Unchanged (already in vocab)
    assert (
        _determine_collision(
            decoded_token="CAT",
            original_token="CAT",
            added_tokens={},
            seen=vocabulary,
            preprocessor=p,
            keep=True,
        )
        == "CAT"
    )
    # Unchanged (already in vocab)
    assert (
        _determine_collision(
            decoded_token="cAT",
            original_token="cAT",
            added_tokens={},
            seen=vocabulary,
            preprocessor=p,
            keep=True,
        )
        == "cAT"
    )
    # Additional test (new token)
    assert (
        _determine_collision(
            decoded_token="SPIN",
            original_token="SPIN",
            added_tokens={},
            seen=vocabulary,
            preprocessor=p,
            keep=True,
        )
        == "spin"
    )
    # Unchanged (already in seen)
    assert (
        _determine_collision(
            decoded_token="SPIN",
            original_token="SPIN",
            added_tokens={},
            seen=vocabulary | {"spin"},
            preprocessor=p,
            keep=True,
        )
        == "SPIN"
    )
    # Unchanged (special token)
    assert (
        _determine_collision(
            decoded_token="Spin",
            original_token="Spin",
            added_tokens={
                "Spin": AddedToken(
                    content="Spin", single_word=True, normalized=True, special=False, lstrip=True, rstrip=True, id=0
                )
            },
            seen=vocabulary,
            preprocessor=p,
            keep=True,
        )
        == "Spin"
    )

    # Byte stuff
    assert (
        _determine_collision(
            decoded_token="dog",
            original_token="dog",
            added_tokens={},
            seen=vocabulary,
            preprocessor=p,
            keep=True,
        )
        == "dog"
    )
    assert (
        _determine_collision(
            decoded_token="DOG",
            original_token="DOG",
            added_tokens={},
            seen=vocabulary,
            preprocessor=p,
            keep=True,
        )
        == "DOG"
    )
    assert (
        _determine_collision(
            decoded_token="cat",
            original_token="cat",
            added_tokens={},
            seen=vocabulary,
            preprocessor=p,
            keep=True,
        )
        == "cat"
    )
    assert (
        _determine_collision(
            decoded_token="CAT",
            original_token="CAT",
            added_tokens={},
            seen=vocabulary,
            preprocessor=p,
            keep=True,
        )
        == "CAT"
    )
    assert (
        _determine_collision(
            decoded_token="cAT",
            original_token="cAT",
            added_tokens={},
            seen=vocabulary,
            preprocessor=p,
            keep=True,
        )
        == "cAT"
    )
    assert (
        _determine_collision(
            decoded_token="SPIN",
            original_token="SPIN",
            added_tokens={},
            seen=vocabulary,
            preprocessor=p,
            keep=True,
        )
        == "spin"
    )
    assert (
        _determine_collision(
            decoded_token="SPIN",
            original_token="SPIN",
            added_tokens={},
            seen=vocabulary | {"spin"},
            preprocessor=p,
            keep=True,
        )
        == "SPIN"
    )
    assert (
        _determine_collision(
            decoded_token="Spin",
            original_token="Spin",
            added_tokens={
                "Spin": AddedToken(
                    content="Spin", single_word=True, normalized=True, special=False, lstrip=True, rstrip=True, id=0
                )
            },
            seen=vocabulary,
            preprocessor=p,
            keep=True,
        )
        == "Spin"
    )

    # Unicode error
    assert (
        _determine_collision(
            original_token="\x80",
            decoded_token="\x80",
            added_tokens={},
            seen=vocabulary,
            preprocessor=p,
            keep=False,
        )
        == "\x80"
    )

    # No unicode error
    assert (
        _determine_collision(
            original_token="\x80",
            decoded_token="\x80",
            added_tokens={},
            seen=vocabulary,
            preprocessor=p,
            keep=False,
        )
        == "\x80"
    )

    # Decode error
    assert (
        _determine_collision(
            original_token="\xa1",
            decoded_token="\xa1",
            added_tokens={},
            seen=vocabulary,
            preprocessor=p,
            keep=False,
        )
        == "\xa1"
    )


def test_decase() -> None:
    """Test the entire decasing procedure."""
    p = Preprocessor(normalizer=Lowercase())
    vocabulary = ["dog", "CAT", "Cat", "cat", "DOG", "Dog", "SPIN"]
    decased = clean_vocabulary(
        vocabulary,
        added_tokens=[],
        preprocessor=p,
        keep=False,
    )
    assert decased == ["dog", None, None, "cat", None, None, None]

    vocabulary = ["dog", "CAT", "Cat", "cat", "DOG", "Dog", "SPIN"]
    decased = clean_vocabulary(
        vocabulary,
        added_tokens=[
            AddedToken(content="SPIN", single_word=True, normalized=True, special=False, lstrip=True, rstrip=True, id=0)
        ],
        preprocessor=p,
        keep=False,
    )
    assert decased == ["dog", None, None, "cat", None, None, "SPIN"]


def test_pretokenizer_preprocessor_splitdigits() -> None:
    """Test whether a pretokenizer that splits sets tokens to None."""
    p = Preprocessor(pretokenizer=Digits(individual_digits=True))

    # This is split into digits
    assert (
        _determine_collision(
            decoded_token="010",
            original_token="010",
            added_tokens={},
            seen=set(),
            preprocessor=p,
            keep=False,
        )
        == None
    )

    # If keep is true, we don't throw anything away
    assert (
        _determine_collision(
            decoded_token="010",
            original_token="010",
            added_tokens={},
            seen=set(),
            preprocessor=p,
            keep=True,
        )
        == "010"
    )

    # Nothing matters if the pretokenizer rejects a token.
    assert (
        _determine_collision(
            decoded_token="010",
            original_token="010",
            added_tokens={},
            seen={"010"},
            preprocessor=p,
            keep=False,
        )
        == None
    )


def test_pretokenizer_digit_normalizer() -> None:
    """Test whether a pretokenizer that splits sets tokens to None."""
    p = Preprocessor(normalizer=Replace(pattern=Regex(r"\d"), content="0"))

    # This is split into digits
    assert (
        _determine_collision(
            decoded_token="123",
            original_token="123",
            added_tokens={},
            seen=set(),
            preprocessor=p,
            keep=False,
        )
        == None
    )

    # If keep is true, we don't throw anything away
    assert (
        _determine_collision(
            decoded_token="123",
            original_token="123",
            added_tokens={},
            seen=set(),
            preprocessor=p,
            keep=True,
        )
        == "000"
    )

    assert (
        _determine_collision(
            decoded_token="010",
            original_token="010",
            added_tokens={},
            seen={"000"},
            preprocessor=p,
            keep=True,
        )
        == "010"
    )

    assert (
        _determine_collision(
            decoded_token="010",
            original_token="010",
            added_tokens={},
            seen=set("010"),
            preprocessor=p,
            keep=False,
        )
        == None
    )


def test_determine_collision_empty_token() -> None:
    """Test whether an empty token is removed."""
    p = Preprocessor(normalizer=Replace(pattern=Regex(r"\d"), content="0"))
    assert (
        _determine_collision(
            decoded_token="",
            original_token="",
            added_tokens={},
            seen=set("010"),
            preprocessor=p,
            keep=False,
        )
        == ""
    )


def test_determine_collision_unknown_token() -> None:
    """Test whether an empty token is removed."""
    p = Preprocessor(normalizer=Replace(pattern=Regex(r"\d"), content="0"))
    assert (
        _determine_collision(
            decoded_token="��",
            original_token="A&",
            added_tokens={},
            seen=set("010"),
            preprocessor=p,
            keep=False,
        )
        == "A&"
    )


def test_determine_decoder() -> None:
    """Test whether an empty token is removed."""
    p = Preprocessor(subword_prefix="##")
    vocabulary = ["dog", "##cat", "cat"]
    decased = clean_vocabulary(
        vocabulary,
        added_tokens=[],
        preprocessor=p,
        keep=False,
    )
    assert decased == ["dog", "##cat", "cat"]
