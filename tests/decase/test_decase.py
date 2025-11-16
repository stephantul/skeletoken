from skeletoken.decase.decase import _determine_collision, decase_vocabulary


def test_determine_collision() -> None:
    """Test the collision detection in the vocabulary."""
    vocabulary = {"dog", "CAT", "Cat", "cat"}
    # Unchanged (already in vocab)
    assert (
        _determine_collision(
            token="dog", is_byte=False, vocab=set(vocabulary), special_tokens=[], seen=set(), lower=False
        )
        == "dog"
    )
    # Unchanged (already in vocab)
    assert (
        _determine_collision(
            token="DOG", is_byte=False, vocab=set(vocabulary), special_tokens=[], seen=set(), lower=False
        )
        == None
    )
    # Unchanged (already lowered)
    assert (
        _determine_collision(
            token="cat", is_byte=False, vocab=set(vocabulary), special_tokens=[], seen=set(), lower=False
        )
        == "cat"
    )
    # Unchanged (already in vocab)
    assert (
        _determine_collision(
            token="CAT", is_byte=False, vocab=set(vocabulary), special_tokens=[], seen=set(), lower=False
        )
        == None
    )
    # Unchanged (already in vocab)
    assert (
        _determine_collision(
            token="cAT", is_byte=False, vocab=set(vocabulary), special_tokens=[], seen=set(), lower=False
        )
        == None
    )
    # Additional test (new token)
    assert (
        _determine_collision(
            token="SPIN", is_byte=False, vocab=set(vocabulary), special_tokens=[], seen=set(), lower=False
        )
        == None
    )
    # Unchanged (already in seen)
    assert (
        _determine_collision(
            token="SPIN",
            is_byte=False,
            vocab=set(vocabulary),
            special_tokens=[],
            seen={"spin"},
            lower=False,
        )
        == None
    )
    # Unchanged (special token)
    assert (
        _determine_collision(
            token="Spin",
            is_byte=False,
            vocab=set(vocabulary),
            special_tokens=["Spin"],
            seen=set(),
            lower=False,
        )
        == "Spin"
    )

    # Byte stuff
    assert (
        _determine_collision(
            token="dog", is_byte=True, vocab=set(vocabulary), special_tokens=[], seen=set(), lower=False
        )
        == "dog"
    )
    assert (
        _determine_collision(
            token="DOG", is_byte=True, vocab=set(vocabulary), special_tokens=[], seen=set(), lower=False
        )
        == None
    )
    assert (
        _determine_collision(
            token="cat", is_byte=True, vocab=set(vocabulary), special_tokens=[], seen=set(), lower=False
        )
        == "cat"
    )
    assert (
        _determine_collision(
            token="CAT", is_byte=True, vocab=set(vocabulary), special_tokens=[], seen=set(), lower=False
        )
        == None
    )
    assert (
        _determine_collision(
            token="cAT", is_byte=True, vocab=set(vocabulary), special_tokens=[], seen=set(), lower=False
        )
        == None
    )
    assert (
        _determine_collision(
            token="SPIN", is_byte=True, vocab=set(vocabulary), special_tokens=[], seen=set(), lower=False
        )
        == None
    )
    assert (
        _determine_collision(
            token="SPIN", is_byte=True, vocab=set(vocabulary), special_tokens=[], seen={"spin"}, lower=False
        )
        == None
    )
    assert (
        _determine_collision(
            token="Spin",
            is_byte=True,
            vocab=set(vocabulary),
            special_tokens=["Spin"],
            seen=set(),
            lower=False,
        )
        == "Spin"
    )

    # Unicode error
    assert (
        _determine_collision(
            token="\x80", is_byte=True, vocab=set(vocabulary), special_tokens=[], seen=set(), lower=False
        )
        == "\x80"
    )

    # No unicode error
    assert (
        _determine_collision(
            token="\x80", is_byte=False, vocab=set(vocabulary), special_tokens=[], seen=set(), lower=False
        )
        == "\x80"
    )

    # Decode error
    assert (
        _determine_collision(
            token="\xa1", is_byte=True, vocab=set(vocabulary), special_tokens=[], seen=set(), lower=False
        )
        == "\xa1"
    )


def test_determine_collision_lower() -> None:
    """Test the collision detection in the vocabulary."""
    vocabulary = {"dog", "CAT", "Cat", "cat"}
    # Unchanged (already in vocab)
    assert (
        _determine_collision(
            token="dog", is_byte=False, vocab=set(vocabulary), special_tokens=[], seen=set(), lower=True
        )
        == "dog"
    )
    # Unchanged (already in vocab)
    assert (
        _determine_collision(
            token="DOG", is_byte=False, vocab=set(vocabulary), special_tokens=[], seen=set(), lower=True
        )
        == None
    )
    # Unchanged (already lowered)
    assert (
        _determine_collision(
            token="cat", is_byte=False, vocab=set(vocabulary), special_tokens=[], seen=set(), lower=True
        )
        == "cat"
    )
    # Unchanged (already in vocab)
    assert (
        _determine_collision(
            token="CAT", is_byte=False, vocab=set(vocabulary), special_tokens=[], seen=set(), lower=True
        )
        == None
    )
    # Unchanged (already in vocab)
    assert (
        _determine_collision(
            token="cAT", is_byte=False, vocab=set(vocabulary), special_tokens=[], seen=set(), lower=True
        )
        == None
    )
    # Additional test (new token)
    assert (
        _determine_collision(
            token="SPIN", is_byte=False, vocab=set(vocabulary), special_tokens=[], seen=set(), lower=True
        )
        == "spin"
    )
    # Unchanged (already in seen)
    assert (
        _determine_collision(
            token="SPIN",
            is_byte=False,
            vocab=set(vocabulary),
            special_tokens=[],
            seen={"spin"},
            lower=True,
        )
        == None
    )
    # Unchanged (special token)
    assert (
        _determine_collision(
            token="Spin",
            is_byte=False,
            vocab=set(vocabulary),
            special_tokens=["Spin"],
            seen=set(),
            lower=True,
        )
        == "Spin"
    )

    # Byte stuff
    assert (
        _determine_collision(
            token="dog", is_byte=True, vocab=set(vocabulary), special_tokens=[], seen=set(), lower=True
        )
        == "dog"
    )
    assert (
        _determine_collision(
            token="DOG", is_byte=True, vocab=set(vocabulary), special_tokens=[], seen=set(), lower=True
        )
        == None
    )
    assert (
        _determine_collision(
            token="cat", is_byte=True, vocab=set(vocabulary), special_tokens=[], seen=set(), lower=True
        )
        == "cat"
    )
    assert (
        _determine_collision(
            token="CAT", is_byte=True, vocab=set(vocabulary), special_tokens=[], seen=set(), lower=True
        )
        == None
    )
    assert (
        _determine_collision(
            token="cAT", is_byte=True, vocab=set(vocabulary), special_tokens=[], seen=set(), lower=True
        )
        == None
    )
    assert (
        _determine_collision(
            token="SPIN", is_byte=True, vocab=set(vocabulary), special_tokens=[], seen=set(), lower=True
        )
        == "spin"
    )
    assert (
        _determine_collision(
            token="SPIN", is_byte=True, vocab=set(vocabulary), special_tokens=[], seen={"spin"}, lower=True
        )
        == None
    )
    assert (
        _determine_collision(
            token="Spin",
            is_byte=True,
            vocab=set(vocabulary),
            special_tokens=["Spin"],
            seen=set(),
            lower=True,
        )
        == "Spin"
    )

    # Unicode error
    assert (
        _determine_collision(
            token="\x80", is_byte=True, vocab=set(vocabulary), special_tokens=[], seen=set(), lower=False
        )
        == "\x80"
    )

    # No unicode error
    assert (
        _determine_collision(
            token="\x80", is_byte=False, vocab=set(vocabulary), special_tokens=[], seen=set(), lower=False
        )
        == "\x80"
    )

    # Decode error
    assert (
        _determine_collision(
            token="\xa1", is_byte=True, vocab=set(vocabulary), special_tokens=[], seen=set(), lower=False
        )
        == "\xa1"
    )


def test_decase() -> None:
    """Test the entire decasing procedure."""
    vocabulary = ["dog", "CAT", "Cat", "cat", "DOG", "Dog", "SPIN"]
    decased = decase_vocabulary(vocabulary, special_tokens=[], is_byte=False, lower=False)
    assert decased == ["dog", None, None, "cat", None, None, None]

    vocabulary = ["dog", "CAT", "Cat", "cat", "DOG", "Dog", "SPIN"]
    decased = decase_vocabulary(vocabulary, special_tokens=["SPIN"], is_byte=False, lower=False)
    assert decased == ["dog", None, None, "cat", None, None, "SPIN"]
