from typing import Literal

from pydantic import BaseModel, ConfigDict

from tokenizerdatamodels.addedtoken import AddedToken
from tokenizerdatamodels.decoders import DecoderDiscriminator
from tokenizerdatamodels.models import ModelDiscriminator
from tokenizerdatamodels.normalizers import NormalizerDiscriminator
from tokenizerdatamodels.postprocessors import PostProcessorDiscriminator
from tokenizerdatamodels.pretokenizers import PreTokenizerDiscriminator


class TokenizerModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    version: Literal["1.0"] = "1.0"
    truncation: None
    padding: None
    added_tokens: list[AddedToken]
    normalizer: None | NormalizerDiscriminator
    pre_tokenizer: None | PreTokenizerDiscriminator
    post_processor: None | PostProcessorDiscriminator
    decoder: None | DecoderDiscriminator
    model: ModelDiscriminator
