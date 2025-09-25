from tokenizers.normalizers import (
    NFD,
    NFKC,
    NFKD,
    BertNormalizer,
    ByteLevel,
    Lowercase,
    Nmt,
    Precompiled,
    Prepend,
    Replace,
    Sequence,
    Strip,
    StripAccents,
)
from tokenizers.normalizers import Normalizer as BaseNormalizer

from skeletoken.normalizers import Normalizer, NormalizerSequence, NormalizerType
from skeletoken.preprocessor.utils import replace_pattern

_class_maping: dict[NormalizerType, type[BaseNormalizer]] = {
    NormalizerType.SEQUENCE: Sequence,
    NormalizerType.LOWERCASE: Lowercase,
    NormalizerType.NFD: NFD,
    NormalizerType.NFC: NFD,
    NormalizerType.NFKD: NFKD,
    NormalizerType.NFKC: NFKC,
    NormalizerType.BYTELEVEL: ByteLevel,
    NormalizerType.NMT: Nmt,
    NormalizerType.PRECOMPILED: Precompiled,
    NormalizerType.STRIP: Strip,
    NormalizerType.STRIPACCENTS: StripAccents,
    NormalizerType.REPLACE: Replace,
    NormalizerType.PREPEND: Prepend,
    NormalizerType.BERTNORMALIZER: BertNormalizer,
}


def create_normalizer(normalizer: Normalizer) -> BaseNormalizer:
    """Create a normalizer from a Normalizer object."""
    t = normalizer.type
    if isinstance(normalizer, NormalizerSequence):
        return Sequence(normalizers=[create_normalizer(child) for child in normalizer.normalizers])  # type: ignore[arg-type]
    selected_class = _class_maping[t]
    dict_obj = normalizer.model_dump()
    dict_obj.pop("type")
    dict_obj = replace_pattern(dict_obj)
    return selected_class(**dict_obj)
