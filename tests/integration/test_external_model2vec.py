import numpy as np
import pytest

pytest.importorskip("model2vec")

from model2vec import StaticModel  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from skeletoken import TokenizerModel  # noqa: E402
from skeletoken.external.model2vec import reshape_embeddings  # noqa: E402

_TOKENIZER_PATH = "tests/data/bert-base-cased"


def _make_static_model(dim: int = 4, seed: int = 0, with_weights: bool = False) -> StaticModel:
    tokenizer = Tokenizer.from_file(_TOKENIZER_PATH)
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(tokenizer.get_vocab_size(), dim)).astype(np.float32)
    weights = rng.normal(size=(tokenizer.get_vocab_size(),)).astype(np.float32) if with_weights else None
    return StaticModel(vectors=vectors, tokenizer=tokenizer, normalize=False, weights=weights)


def test_reshape_embeddings_does_not_mutate_original() -> None:
    """Test that reshape_embeddings leaves the input model's arrays and config untouched.

    config/language/token_mapping are stored by reference in StaticModel.__init__, so without
    copying them, mutating the returned model's config would silently mutate the original's too.
    """
    model = _make_static_model(with_weights=True)
    assert model.weights is not None
    model.config["some_key"] = "some_value"
    model.language = ["en"]
    original_embedding = model.embedding.copy()
    original_weights = model.weights.copy()
    tokenizer_model = TokenizerModel.from_pretrained(_TOKENIZER_PATH)
    added = tokenizer_model.add_token_to_vocabulary("skeletokentesttoken")

    reshaped = reshape_embeddings(model, added)

    assert reshaped is not model
    assert reshaped.config is not model.config
    assert reshaped.language is not model.language
    assert np.array_equal(model.embedding, original_embedding)
    assert np.array_equal(model.weights, original_weights)

    assert reshaped.language is not None
    # Mutating the returned model's config/language must not affect the original.
    reshaped.config["some_key"] = "changed"
    reshaped.language.append("nl")
    assert model.config["some_key"] == "some_value"
    assert model.language == ["en"]


def test_reshape_embeddings_remaps_on_shrink() -> None:
    """Test that decasing (which shrinks the vocabulary) preserves embeddings for surviving tokens."""
    model = _make_static_model()
    tokenizer_model = TokenizerModel.from_pretrained(_TOKENIZER_PATH)
    decased = tokenizer_model.decase_vocabulary()

    reshaped = reshape_embeddings(model, decased)
    assert reshaped.embedding.shape == (decased.vocabulary_size, 4)

    delta = decased.model_delta
    assert len(delta.token_mapping) > 1000
    for new_id, old_id in delta.token_mapping.items():
        assert np.allclose(reshaped.embedding[new_id], model.embedding[old_id])

    assert np.allclose(reshaped.encode(["amsterdam"])[0], reshaped.encode(["Amsterdam"])[0])


def test_reshape_embeddings_new_token_gets_a_row() -> None:
    """Test that a token added after loading gets a real (non-degenerate) embedding row."""
    model = _make_static_model()
    tokenizer_model = TokenizerModel.from_pretrained(_TOKENIZER_PATH)
    added = tokenizer_model.add_token_to_vocabulary("skeletokentesttoken")

    reshaped = reshape_embeddings(model, added)
    assert reshaped.embedding.shape == (added.vocabulary_size, 4) == (tokenizer_model.vocabulary_size + 1, 4)

    vector = reshaped.encode(["skeletokentesttoken"])[0]
    assert vector.shape == (4,)


def test_reshape_embeddings_grows_weights_array() -> None:
    """Growing the vocabulary must also grow `weights`, not just `embedding`."""
    model = _make_static_model(with_weights=True)
    assert model.weights is not None
    tokenizer_model = TokenizerModel.from_pretrained(_TOKENIZER_PATH)
    added = tokenizer_model.add_token_to_vocabulary("skeletokentesttoken")

    reshaped = reshape_embeddings(model, added)
    assert reshaped.weights is not None
    assert reshaped.weights.shape == reshaped.embedding.shape[:1] == (added.vocabulary_size,)

    assert np.allclose(reshaped.weights[:-1], model.weights)
    assert reshaped.weights[-1] == 1.0
    vector = reshaped.encode(["skeletokentesttoken"])[0]
    assert vector.shape == (4,)


def test_reshape_embeddings_remaps_weights_on_shrink() -> None:
    """Test that weights (not just embeddings) get remapped correctly when the vocabulary shrinks."""
    model = _make_static_model(with_weights=True)
    tokenizer_model = TokenizerModel.from_pretrained(_TOKENIZER_PATH)
    decased = tokenizer_model.decase_vocabulary()

    reshaped = reshape_embeddings(model, decased)
    assert model.weights is not None
    assert reshaped.weights is not None
    assert reshaped.weights.shape == (decased.vocabulary_size,)

    delta = decased.model_delta
    for new_id, old_id in delta.token_mapping.items():
        assert reshaped.weights[new_id] == model.weights[old_id]


def test_reshape_embeddings_raises_on_quantized_model() -> None:
    """Test that a mismatched weights/embedding length (a quantized model) raises ValueError."""
    model = _make_static_model(with_weights=True)
    assert model.weights is not None
    model.weights = model.weights[:-5]
    tokenizer_model = TokenizerModel.from_pretrained(_TOKENIZER_PATH)
    added = tokenizer_model.add_token_to_vocabulary("skeletokentesttoken")

    with pytest.raises(ValueError, match="quantized"):
        reshape_embeddings(model, added)
