from tokenizers.normalizers import Lowercase

from skeletoken.addedtoken import AddedToken
from skeletoken.decase.decase import _determine_collision, clean_vocabulary
from skeletoken.preprocessor import Preprocessor


def test_determine_collision() -> None:
    """Test the collision detection in the vocabulary."""
    vocabulary = {"dog", "CAT", "Cat", "cat"}
    # Unchanged (already in vocab)
    p = Preprocessor(normalizer=Lowercase())
    assert (
        _determine_collision(
            token="dog", is_byte=False, vocab=set(vocabulary), added_tokens={}, seen=set(), preprocessor=p, keep=True
        )
        == "dog"
    )
    # Unchanged (already in vocab)
    assert (
        _determine_collision(
            token="DOG", is_byte=False, vocab=set(vocabulary), added_tokens={}, seen=set(), preprocessor=p, keep=False
        )
        == None
    )
    # Unchanged (already lowered)
    assert (
        _determine_collision(
            token="cat", is_byte=False, vocab=set(vocabulary), added_tokens={}, seen=set(), preprocessor=p, keep=False
        )
        == "cat"
    )
    # Unchanged (already in vocab)
    assert (
        _determine_collision(
            token="CAT", is_byte=False, vocab=set(vocabulary), added_tokens={}, seen=set(), preprocessor=p, keep=False
        )
        == None
    )
    # Unchanged (already in vocab)
    assert (
        _determine_collision(
            token="cAT", is_byte=False, vocab=set(vocabulary), added_tokens={}, seen=set(), preprocessor=p, keep=False
        )
        == None
    )
    # Additional test (new token)
    assert (
        _determine_collision(
            token="SPIN", is_byte=False, vocab=set(vocabulary), added_tokens={}, seen=set(), preprocessor=p, keep=False
        )
        == "spin"
    )
    # Unchanged (already in seen)
    assert (
        _determine_collision(
            token="SPIN",
            is_byte=False,
            vocab=set(vocabulary),
            added_tokens={},
            seen={"spin"},
            preprocessor=p,
            keep=False,
        )
        == None
    )
    # Unchanged (special token)
    assert (
        _determine_collision(
            token="Spin",
            is_byte=False,
            vocab=set(vocabulary),
            added_tokens={
                "Spin": AddedToken(
                    content="Spin", single_word=True, normalized=True, special=False, lstrip=True, rstrip=True, id=0
                )
            },
            seen=set(),
            preprocessor=p,
            keep=False,
        )
        == "Spin"
    )

    # Byte stuff
    assert (
        _determine_collision(
            token="dog",
            is_byte=True,
            vocab=set(vocabulary),
            added_tokens={},
            seen=set(),
            preprocessor=p,
            keep=False,
        )
        == "dog"
    )
    assert (
        _determine_collision(
            token="DOG",
            is_byte=True,
            vocab=set(vocabulary),
            added_tokens={},
            seen=set(),
            preprocessor=p,
            keep=False,
        )
        == None
    )
    assert (
        _determine_collision(
            token="cat",
            is_byte=True,
            vocab=set(vocabulary),
            added_tokens={},
            seen=set(),
            preprocessor=p,
            keep=False,
        )
        == "cat"
    )
    assert (
        _determine_collision(
            token="CAT",
            is_byte=True,
            vocab=set(vocabulary),
            added_tokens={},
            seen=set(),
            preprocessor=p,
            keep=False,
        )
        == None
    )
    assert (
        _determine_collision(
            token="cAT",
            is_byte=True,
            vocab=set(vocabulary),
            added_tokens={},
            seen=set(),
            preprocessor=p,
            keep=False,
        )
        == None
    )
    assert (
        _determine_collision(
            token="SPIN",
            is_byte=True,
            vocab=set(vocabulary),
            added_tokens={},
            seen=set(),
            preprocessor=p,
            keep=False,
        )
        == "spin"
    )
    assert (
        _determine_collision(
            token="SPIN",
            is_byte=True,
            vocab=set(vocabulary),
            added_tokens={},
            seen={"spin"},
            preprocessor=p,
            keep=False,
        )
        == None
    )
    assert (
        _determine_collision(
            token="Spin",
            is_byte=True,
            vocab=set(vocabulary),
            added_tokens={
                "Spin": AddedToken(
                    content="Spin", single_word=True, normalized=True, special=False, lstrip=True, rstrip=True, id=0
                )
            },
            seen=set(),
            preprocessor=p,
            keep=False,
        )
        == "Spin"
    )

    # Unicode error
    assert (
        _determine_collision(
            token="\x80",
            is_byte=True,
            vocab=set(vocabulary),
            added_tokens={},
            seen=set(),
            preprocessor=p,
            keep=False,
        )
        == "\x80"
    )

    # No unicode error
    assert (
        _determine_collision(
            token="\x80",
            is_byte=False,
            vocab=set(vocabulary),
            added_tokens={},
            seen=set(),
            preprocessor=p,
            keep=False,
        )
        == "\x80"
    )

    # Decode error
    assert (
        _determine_collision(
            token="\xa1",
            is_byte=True,
            vocab=set(vocabulary),
            added_tokens={},
            seen=set(),
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
            token="dog",
            is_byte=False,
            vocab=set(vocabulary),
            added_tokens={},
            seen=set(),
            preprocessor=p,
            keep=True,
        )
        == "dog"
    )
    # Unchanged (already in vocab)
    assert (
        _determine_collision(
            token="DOG",
            is_byte=False,
            vocab=set(vocabulary),
            added_tokens={},
            seen=set(),
            preprocessor=p,
            keep=False,
        )
        == None
    )
    # Unchanged (already lowered)
    assert (
        _determine_collision(
            token="cat",
            is_byte=False,
            vocab=set(vocabulary),
            added_tokens={},
            seen=set(),
            preprocessor=p,
            keep=True,
        )
        == "cat"
    )
    # Unchanged (already in vocab)
    assert (
        _determine_collision(
            token="CAT",
            is_byte=False,
            vocab=set(vocabulary),
            added_tokens={},
            seen=set(),
            preprocessor=p,
            keep=True,
        )
        == "CAT"
    )
    # Unchanged (already in vocab)
    assert (
        _determine_collision(
            token="cAT",
            is_byte=False,
            vocab=set(vocabulary),
            added_tokens={},
            seen=set(),
            preprocessor=p,
            keep=True,
        )
        == "cAT"
    )
    # Additional test (new token)
    assert (
        _determine_collision(
            token="SPIN",
            is_byte=False,
            vocab=set(vocabulary),
            added_tokens={},
            seen=set(),
            preprocessor=p,
            keep=True,
        )
        == "spin"
    )
    # Unchanged (already in seen)
    assert (
        _determine_collision(
            token="SPIN",
            is_byte=False,
            vocab=set(vocabulary),
            added_tokens={},
            seen={"spin"},
            preprocessor=p,
            keep=True,
        )
        == "SPIN"
    )
    # Unchanged (special token)
    assert (
        _determine_collision(
            token="Spin",
            is_byte=False,
            vocab=set(vocabulary),
            added_tokens={
                "Spin": AddedToken(
                    content="Spin", single_word=True, normalized=True, special=False, lstrip=True, rstrip=True, id=0
                )
            },
            seen=set(),
            preprocessor=p,
            keep=True,
        )
        == "Spin"
    )

    # Byte stuff
    assert (
        _determine_collision(
            token="dog",
            is_byte=True,
            vocab=set(vocabulary),
            added_tokens={},
            seen=set(),
            preprocessor=p,
            keep=True,
        )
        == "dog"
    )
    assert (
        _determine_collision(
            token="DOG",
            is_byte=True,
            vocab=set(vocabulary),
            added_tokens={},
            seen=set(),
            preprocessor=p,
            keep=True,
        )
        == "DOG"
    )
    assert (
        _determine_collision(
            token="cat",
            is_byte=True,
            vocab=set(vocabulary),
            added_tokens={},
            seen=set(),
            preprocessor=p,
            keep=True,
        )
        == "cat"
    )
    assert (
        _determine_collision(
            token="CAT",
            is_byte=True,
            vocab=set(vocabulary),
            added_tokens={},
            seen=set(),
            preprocessor=p,
            keep=True,
        )
        == "CAT"
    )
    assert (
        _determine_collision(
            token="cAT",
            is_byte=True,
            vocab=set(vocabulary),
            added_tokens={},
            seen=set(),
            preprocessor=p,
            keep=True,
        )
        == "cAT"
    )
    assert (
        _determine_collision(
            token="SPIN",
            is_byte=True,
            vocab=set(vocabulary),
            added_tokens={},
            seen=set(),
            preprocessor=p,
            keep=True,
        )
        == "spin"
    )
    assert (
        _determine_collision(
            token="SPIN",
            is_byte=True,
            vocab=set(vocabulary),
            added_tokens={},
            seen={"spin"},
            preprocessor=p,
            keep=True,
        )
        == "SPIN"
    )
    assert (
        _determine_collision(
            token="Spin",
            is_byte=True,
            vocab=set(vocabulary),
            added_tokens={
                "Spin": AddedToken(
                    content="Spin", single_word=True, normalized=True, special=False, lstrip=True, rstrip=True, id=0
                )
            },
            seen=set(),
            preprocessor=p,
            keep=True,
        )
        == "Spin"
    )

    # Unicode error
    assert (
        _determine_collision(
            token="\x80",
            is_byte=True,
            vocab=set(vocabulary),
            added_tokens={},
            seen=set(),
            preprocessor=p,
            keep=False,
        )
        == "\x80"
    )

    # No unicode error
    assert (
        _determine_collision(
            token="\x80",
            is_byte=False,
            vocab=set(vocabulary),
            added_tokens={},
            seen=set(),
            preprocessor=p,
            keep=False,
        )
        == "\x80"
    )

    # Decode error
    assert (
        _determine_collision(
            token="\xa1",
            is_byte=True,
            vocab=set(vocabulary),
            added_tokens={},
            seen=set(),
            preprocessor=p,
            keep=False,
        )
        == "\xa1"
    )


def test_decase() -> None:
    """Test the entire decasing procedure."""
    p = Preprocessor(normalizer=Lowercase())
    vocabulary = ["dog", "CAT", "Cat", "cat", "DOG", "Dog", "SPIN"]
    decased = clean_vocabulary(vocabulary, added_tokens=[], is_byte=False, preprocessor=p, keep=False)
    assert decased == ["dog", None, None, "cat", None, None, "spin"]

    vocabulary = ["dog", "CAT", "Cat", "cat", "DOG", "Dog", "SPIN"]
    decased = clean_vocabulary(
        vocabulary,
        added_tokens=[
            AddedToken(content="SPIN", single_word=True, normalized=True, special=False, lstrip=True, rstrip=True, id=0)
        ],
        is_byte=False,
        preprocessor=p,
        keep=False,
    )
    assert decased == ["dog", None, None, "cat", None, None, "SPIN"]
