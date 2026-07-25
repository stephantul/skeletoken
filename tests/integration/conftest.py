import pytest

_TOKENIZER_PATH = "tests/data/bert-base-cased"


@pytest.fixture(scope="session")
def small_bert_checkpoint_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Build a small offline BERT checkpoint whose vocabulary matches `_TOKENIZER_PATH`."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    tokenizer = Tokenizer.from_file(_TOKENIZER_PATH)
    config = BertConfig(
        vocab_size=tokenizer.get_vocab_size(),
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=8,
        max_position_embeddings=32,
        pad_token_id=0,
    )
    torch.manual_seed(0)
    model = BertModel(config)
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )

    checkpoint_dir = tmp_path_factory.mktemp("tiny_bert_checkpoint")
    model.save_pretrained(checkpoint_dir)
    fast_tokenizer.save_pretrained(checkpoint_dir)
    return str(checkpoint_dir)
