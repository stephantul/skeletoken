from skeletoken.decase.decase import _determine_collision, decase_vocabulary


def test_determine_collision() -> None:
    """Test the collision detection in the vocabulary."""
    vocabulary = {"dog", "CAT", "Cat", "cat"}
    # Unchanged (already in vocab)
    assert (
        _determine_collision(token="dog", is_byte=False, vocab=set(vocabulary), special_tokens=[], seen=set()) == "dog"
    )
    # Unchanged (already in vocab)
    assert (
        _determine_collision(token="DOG", is_byte=False, vocab=set(vocabulary), special_tokens=[], seen=set()) == "DOG"
    )
    # Unchanged (already lowered)
    assert (
        _determine_collision(token="cat", is_byte=False, vocab=set(vocabulary), special_tokens=[], seen=set()) == "cat"
    )
    # Unchanged (already in vocab)
    assert (
        _determine_collision(token="CAT", is_byte=False, vocab=set(vocabulary), special_tokens=[], seen=set()) == "CAT"
    )
    # Unchanged (already in vocab)
    assert (
        _determine_collision(token="cAT", is_byte=False, vocab=set(vocabulary), special_tokens=[], seen=set()) == "cAT"
    )
    # Additional test (new token)
    assert (
        _determine_collision(token="SPIN", is_byte=False, vocab=set(vocabulary), special_tokens=[], seen=set())
        == "spin"
    )
    # Unchanged (already in seen)
    assert (
        _determine_collision(token="SPIN", is_byte=False, vocab=set(vocabulary), special_tokens=[], seen={"spin"})
        == "SPIN"
    )
    # Unchanged (special token)
    assert (
        _determine_collision(token="Spin", is_byte=False, vocab=set(vocabulary), special_tokens=["Spin"], seen=set())
        == "Spin"
    )

    # Byte stuff
    assert (
        _determine_collision(token="dog", is_byte=True, vocab=set(vocabulary), special_tokens=[], seen=set()) == "dog"
    )
    assert (
        _determine_collision(token="DOG", is_byte=True, vocab=set(vocabulary), special_tokens=[], seen=set()) == "DOG"
    )
    assert (
        _determine_collision(token="cat", is_byte=True, vocab=set(vocabulary), special_tokens=[], seen=set()) == "cat"
    )
    assert (
        _determine_collision(token="CAT", is_byte=True, vocab=set(vocabulary), special_tokens=[], seen=set()) == "CAT"
    )
    assert (
        _determine_collision(token="cAT", is_byte=True, vocab=set(vocabulary), special_tokens=[], seen=set()) == "cAT"
    )
    assert (
        _determine_collision(token="SPIN", is_byte=True, vocab=set(vocabulary), special_tokens=[], seen=set()) == "spin"
    )
    assert (
        _determine_collision(token="SPIN", is_byte=True, vocab=set(vocabulary), special_tokens=[], seen={"spin"})
        == "SPIN"
    )
    assert (
        _determine_collision(token="Spin", is_byte=True, vocab=set(vocabulary), special_tokens=["Spin"], seen=set())
        == "Spin"
    )

    # Unicode error
    assert (
        _determine_collision(token="\x80", is_byte=True, vocab=set(vocabulary), special_tokens=[], seen=set()) == "\x80"
    )

    # No unicode error
    assert (
        _determine_collision(token="\x80", is_byte=False, vocab=set(vocabulary), special_tokens=[], seen=set())
        == "\x80"
    )

    # Decode error
    assert (
        _determine_collision(token="\xa1", is_byte=True, vocab=set(vocabulary), special_tokens=[], seen=set()) == "\xa1"
    )


def test_decase() -> None:
    """Test the entire decasing procedure."""
    vocabulary = ["dog", "CAT", "Cat", "cat", "DOG", "Dog", "SPIN"]
    decased = decase_vocabulary(vocabulary, special_tokens=[], is_byte=False)
    assert decased == ["dog", "CAT", "Cat", "cat", "DOG", "Dog", "spin"]

    vocabulary = ["dog", "CAT", "Cat", "cat", "DOG", "Dog", "SPIN"]
    decased = decase_vocabulary(vocabulary, special_tokens=["SPIN"], is_byte=False)
    assert decased == ["dog", "CAT", "Cat", "cat", "DOG", "Dog", "SPIN"]
