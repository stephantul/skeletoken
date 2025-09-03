import pytest

from skeletoken.vocabulary import UnigramVocabulary, Vocabulary


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
