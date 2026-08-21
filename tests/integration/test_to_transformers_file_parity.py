"""Compare the files a `to_transformers()` tokenizer writes on disk against the source."""

import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from skeletoken import TokenizerModel

_REPOS_WITH_CONFIG = ["bert-base-uncased", "gpt2", "ModernBERT-base", "multilingual-e5-base"]


@pytest.mark.parametrize("repo", _REPOS_WITH_CONFIG)
def test_saved_tokenizer_json_matches_to_tokenizer(repo: str, tmp_path: Path) -> None:
    """`tokenizer.json` written via `to_transformers().save_pretrained()` matches `to_tokenizer().save()`, ignoring `padding`."""
    path = Path("tests/data") / repo
    model = TokenizerModel.from_pretrained(str(path))

    model.to_tokenizer().save(str(tmp_path / "direct.json"))
    model.to_transformers().save_pretrained(tmp_path / "via_transformers")

    direct = json.loads((tmp_path / "direct.json").read_text())
    via_transformers = json.loads((tmp_path / "via_transformers" / "tokenizer.json").read_text())
    direct.pop("padding", None)
    via_transformers.pop("padding", None)
    assert via_transformers == direct


@pytest.mark.parametrize("repo", _REPOS_WITH_CONFIG)
def test_saved_tokenizer_config_json_matches_control_resave(repo: str, tmp_path: Path) -> None:
    """`tokenizer_config.json` from a skeletoken roundtrip vs. the same transformers version resaving the original."""
    path = Path("tests/data") / repo

    # AutoTokenizer, not bare PreTrainedTokenizerFast: needs to resolve the same class as model._original_class.
    control_tokenizer = AutoTokenizer.from_pretrained(str(path))
    control_dir = tmp_path / "control"
    control_tokenizer.save_pretrained(control_dir)
    control = json.loads((control_dir / "tokenizer_config.json").read_text())

    model = TokenizerModel.from_pretrained(str(path))
    skeletoken_dir = tmp_path / "skeletoken"
    model.to_transformers().save_pretrained(skeletoken_dir)
    roundtripped = json.loads((skeletoken_dir / "tokenizer_config.json").read_text())

    # Not modeled (max_length, is_local), or verified elsewhere (add_prefix_space).
    for key in ("max_length", "is_local", "add_prefix_space"):
        control.pop(key, None)
        roundtripped.pop(key, None)
    assert roundtripped == control
