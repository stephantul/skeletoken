import pytest

from skeletoken.vocabulary import UnigramVocabulary, Vocabulary, tokens_ordered_by_id


def _simple_vocabulary_fixture() -> Vocabulary:
    """Simple pseudofixture to be used in a parametrize."""
    return Vocabulary({"hello": 0, "world": 1})


def _simple_unigram_fixture() -> UnigramVocabulary:
    """Simple pseudofixture to be used in a parametrize."""
    return UnigramVocabulary([("hello", -0.1), ("world", -0.2)])


@pytest.mark.parametrize("vocab", [_simple_vocabulary_fixture(), _simple_unigram_fixture()])
def test_vocabulary(vocab: Vocabulary | UnigramVocabulary) -> None:
    """Test the vocabulary implementation."""
    assert vocab.vocabulary == {"hello": 0, "world": 1}
    assert vocab.sorted_vocabulary == ["hello", "world"]


@pytest.mark.parametrize("vocab", [_simple_vocabulary_fixture(), _simple_unigram_fixture()])
def test_add_token(vocab: Vocabulary | UnigramVocabulary) -> None:
    """Test the vocabulary implementation."""
    vocab.add_token("new_token")
    assert vocab.vocabulary["new_token"] == len(vocab.vocabulary) - 1
    assert vocab.sorted_vocabulary == ["hello", "world", "new_token"]
    with pytest.raises(ValueError):
        vocab.add_token("new_token")


@pytest.mark.parametrize("vocab", [_simple_vocabulary_fixture(), _simple_unigram_fixture()])
def test_remove_token(vocab: Vocabulary | UnigramVocabulary) -> None:
    """Test the vocabulary implementation."""
    vocab.add_token("new_token")
    assert vocab.vocabulary["new_token"] == len(vocab.vocabulary) - 1
    assert vocab.sorted_vocabulary == ["hello", "world", "new_token"]
    vocab.remove_token("new_token")
    assert "new_token" not in vocab.vocabulary
    assert vocab.sorted_vocabulary == ["hello", "world"]
    with pytest.raises(ValueError):
        vocab.remove_token("new_token")


@pytest.mark.parametrize("vocab", [_simple_vocabulary_fixture(), _simple_unigram_fixture()])
def test_remove_tokens(vocab: Vocabulary | UnigramVocabulary) -> None:
    """Test the vocabulary implementation."""
    vocab.add_token("new_token")
    assert vocab.vocabulary["new_token"] == len(vocab.vocabulary) - 1
    assert vocab.sorted_vocabulary == ["hello", "world", "new_token"]
    vocab.remove_tokens(vocab.sorted_vocabulary)
    assert vocab.sorted_vocabulary == []


@pytest.mark.parametrize("vocab", [_simple_vocabulary_fixture(), _simple_unigram_fixture()])
def test_replace_token(vocab: Vocabulary | UnigramVocabulary) -> None:
    """Test the vocabulary implementation."""
    vocab.add_token("new_token")
    assert vocab.vocabulary["new_token"] == len(vocab.vocabulary) - 1
    assert vocab.sorted_vocabulary == ["hello", "world", "new_token"]
    vocab.replace_token("new_token", "another_token")
    assert "new_token" not in vocab.vocabulary
    assert vocab.vocabulary["another_token"] == len(vocab.vocabulary) - 1
    assert vocab.sorted_vocabulary == ["hello", "world", "another_token"]
    with pytest.raises(ValueError):
        vocab.replace_token("new_token", "another_token")
    with pytest.raises(ValueError):
        vocab.replace_token("hello", "hello")


@pytest.mark.parametrize("vocab", [_simple_vocabulary_fixture(), _simple_unigram_fixture()])
def test_replace_vocabulary(vocab: Vocabulary | UnigramVocabulary) -> None:
    """Test the vocabulary implementation."""
    assert vocab.sorted_vocabulary == ["hello", "world"]
    vocab.replace_vocabulary(["new_token", "hello"])
    assert vocab.sorted_vocabulary == ["new_token", "hello"]

    if isinstance(vocab, UnigramVocabulary):
        with pytest.raises(ValueError):
            vocab.replace_vocabulary(["a", "b", "c"])
    else:
        # Should not raise for regular vocabulary
        vocab.replace_vocabulary(["only_one_token"])

    vocab.replace_vocabulary(["a", None])
    assert vocab.sorted_vocabulary == ["a"]


@pytest.mark.parametrize("vocab", [_simple_vocabulary_fixture(), _simple_unigram_fixture()])
def test_in(vocab: Vocabulary | UnigramVocabulary) -> None:
    """Test the vocabulary implementation."""
    assert "hello" in vocab
    assert "world" in vocab
    assert "new_token" not in vocab


@pytest.mark.parametrize("vocab", [_simple_vocabulary_fixture(), _simple_unigram_fixture()])
def test_getitem(vocab: Vocabulary | UnigramVocabulary) -> None:
    """Test the vocabulary implementation."""
    assert vocab["hello"] == 0
    assert vocab["world"] == 1
    with pytest.raises(KeyError):
        vocab["new_token"]


@pytest.mark.parametrize("vocab", [_simple_vocabulary_fixture(), _simple_unigram_fixture()])
def test_len(vocab: Vocabulary | UnigramVocabulary) -> None:
    """Test the vocabulary implementation."""
    assert len(vocab) == 2


def test_inverse_vocabulary_with_gap() -> None:
    """`inverse_vocabulary` must look tokens up by their real ID, not by list position."""
    vocab = Vocabulary({"a": 0, "b": 2, "c": 3})
    assert vocab.inverse_vocabulary == {0: "a", 2: "b", 3: "c"}


def test_add_token_reuses_gap_instead_of_colliding() -> None:
    """`add_token` must not silently collide with an existing ID when there's a gap.

    len(self.root) is only a safe ID to assign when IDs are already contiguous. If a
    token was removed leaving a gap, len(self.root) can point at an ID that's still in
    use by another token.
    """
    vocab = Vocabulary({"a": 0, "b": 1, "c": 3})  # gap at id 2, len(root) == 3 collides with "c"
    vocab.add_token("d")
    assert vocab.vocabulary["d"] == 2
    assert len(set(vocab.vocabulary.values())) == len(vocab.vocabulary)


def test_sorted_vocabulary_raises_on_gap() -> None:
    """`sorted_vocabulary` must fail loud rather than silently mis-index a gapped vocabulary."""
    vocab = Vocabulary({"a": 0, "b": 2, "c": 3})
    with pytest.raises(ValueError, match="gaps"):
        _ = vocab.sorted_vocabulary


def test_tokens_ordered_by_id_allows_gaps() -> None:
    """Unlike `sorted_vocabulary`, `tokens_ordered_by_id` must not raise on a gapped vocabulary."""
    vocab = Vocabulary({"a": 0, "b": 2, "c": 3})
    assert tokens_ordered_by_id(vocab.inverse_vocabulary) == ["a", "b", "c"]
