from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from skeletoken.common import PathLike
from skeletoken.decoders import add_clean_up_tokenization_spaces, strip_clean_up_tokenization_spaces
from skeletoken.normalizers import add_prepend_normalizer
from skeletoken.padding import is_basic_padding, to_transformers_padding_kwargs
from skeletoken.post_processors import TemplatePostProcessor
from skeletoken.pre_tokenizers import already_adds_prefix_space
from skeletoken.truncation import Truncation, TruncationDirection, TruncationStrategy

if TYPE_CHECKING:
    from skeletoken.base import TokenizerModel

logger = logging.getLogger(__name__)

# transformers uses this as the sentinel for "model_max_length was never set".
_UNSET_MODEL_MAX_LENGTH = int(1e15)


def convert_transformers_tokenizer_to_model(  # noqa: C901  # Just complicated.
    cls: type[TokenizerModel], hf_tokenizer: PreTrainedTokenizerFast
) -> TokenizerModel:
    """Load a HuggingFace tokenizer from a local path or a model repo."""
    special_tokens = hf_tokenizer.special_tokens_map
    unk_token = special_tokens.get("unk_token", None)
    pad_token = special_tokens.get("pad_token", None)

    model = cls.from_tokenizer(hf_tokenizer.backend_tokenizer)

    if getattr(hf_tokenizer, "clean_up_tokenization_spaces", False):
        model.decoder = add_clean_up_tokenization_spaces(model.decoder)

    if getattr(hf_tokenizer, "add_prefix_space", False) and not already_adds_prefix_space(model.pre_tokenizer):
        model.normalizer = add_prepend_normalizer(model.normalizer, " ")

    if getattr(hf_tokenizer, "_should_update_post_processor", False):
        post_processor = model.post_processor
        if isinstance(post_processor, TemplatePostProcessor) and not post_processor.special_tokens:
            logger.warning(
                "The HuggingFace tokenizer had no post-processor, but transformers synthesized a "
                "no-op TemplateProcessing post-processor when loading it (a change in transformers>=5). "
                "Resetting Skeletoken's post_processor to None to match the original tokenizer."
            )
            model.post_processor = None

    if unk_token is not None and isinstance(unk_token, str):
        if model.unk_token is not None and model.unk_token != unk_token:
            logger.warning(
                f"Overriding existing unk_token '{model.unk_token}' with the one from "
                f"the HuggingFace tokenizer: '{unk_token}'."
            )
        if model.unk_token is None:
            logger.warning(
                "HuggingFace tokenizer defines an unk_token, but the Skeletoken model does not. "
                f"Setting it to '{unk_token}'."
            )
        model.unk_token = unk_token
    if pad_token is not None and isinstance(pad_token, str):
        if model.pad_token is not None and model.pad_token != pad_token:
            logger.warning(
                f"Overriding existing pad_token '{model.pad_token}' "
                f"with the one from the HuggingFace tokenizer: '{pad_token}'."
            )
        if model.pad_token is None:
            logger.warning(
                "HuggingFace tokenizer defines a pad_token, but the Skeletoken model does not. "
                f"Setting it to '{pad_token}'."
            )
        model.pad_token = pad_token

    model_max_length = hf_tokenizer.model_max_length
    if model_max_length < _UNSET_MODEL_MAX_LENGTH:
        if model.truncation is None:
            model.truncation = Truncation(
                direction=TruncationDirection.RIGHT,
                max_length=model_max_length,
                strategy=TruncationStrategy.LONGEST_FIRST,
                stride=0,
            )
        else:
            model.truncation.max_length = model_max_length

    model._original_class = type(hf_tokenizer)
    return model


def load_transformers_model(cls: type[TokenizerModel], path: PathLike) -> TokenizerModel:  # pragma: nocover
    """Load a HuggingFace tokenizer from a local path or a model repo."""
    # transformers>=5's AutoTokenizer.from_pretrained stub returns TokenizersBackend |
    # SentencePieceBackend rather than PreTrainedTokenizerFast, even though
    # PreTrainedTokenizerFast is literally an alias for TokenizersBackend at runtime.
    # The mismatch (and whether it fires at all) depends on the installed transformers version.
    hf_tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(path)  # type: ignore[assignment]
    return convert_transformers_tokenizer_to_model(cls, hf_tokenizer)


def convert_model_to_transformers_tokenizer(
    model: TokenizerModel, tokenizer_class: type[PreTrainedTokenizerFast] | None = None
) -> PreTrainedTokenizerFast:
    """Convert a TokenizerModel to a HuggingFace tokenizer."""
    model = model.deep_copy()
    pad_token = model.pad_token
    padding = model.padding
    if is_basic_padding(model.padding):
        # Unset the padding so it isn't baked into the tokenizer we hand to transformers.
        model.padding = None
    original_decoder = model.decoder
    model.decoder = strip_clean_up_tokenization_spaces(model.decoder)
    clean_up_tokenization_spaces = model.decoder != original_decoder
    tokenizer = model.to_tokenizer()
    if tokenizer_class is None:
        if model._original_class is not None:
            tokenizer_class = model._original_class
        else:
            tokenizer_class = PreTrainedTokenizerFast
    padding_kwargs = to_transformers_padding_kwargs(padding)
    add_prefix_space = model.adds_prefix_space
    prefix_space_kwargs = {} if add_prefix_space is None else {"add_prefix_space": add_prefix_space}
    tok = tokenizer_class(
        tokenizer_object=tokenizer,
        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        **padding_kwargs,
        **prefix_space_kwargs,
    )
    if model.truncation is not None:
        tok.model_max_length = model.truncation.max_length
    tok.pad_token = pad_token
    tok.unk_token = model.unk_token
    if padding is not None:
        # transformers doesn't copy these from init_kwargs onto live attributes.
        tok.pad_to_multiple_of = padding.pad_to_multiple_of
        tok._pad_token_type_id = padding.pad_type_id
    if model.bos:
        if len(model.bos) > 1:
            logger.warning(f"Tokenizer has multiple bos tokens: {model.bos}. Not setting it automatically.")
        else:
            tok.bos_token = model.bos[0]
    if model.eos:
        if len(model.eos) > 1:
            logger.warning(f"Tokenizer has multiple eos tokens: {model.eos}. Not setting it automatically.")
        else:
            tok.eos_token = model.eos[0]

    return tok
