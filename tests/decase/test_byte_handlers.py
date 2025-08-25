from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import ByteLevel

from skeletoken.decase.byte_handlers import text_to_token_str, token_to_bytes


def test_token_to_bytes() -> None:
    """Test the conversion from token to bytes."""
    assert token_to_bytes("abba") == b"abba"
    assert token_to_bytes("Ġabba") == b" abba"
    assert token_to_bytes("ãģĵãĤĵãģ«ãģ¡ãģ¯") == b"\xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf"


def test_to_token_str() -> None:
    """Test the conversion from bytes to token string."""
    assert text_to_token_str("abba") == "abba"
    assert text_to_token_str(" abba") == "Ġabba"
    bytes = b"\xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf".decode("utf-8")
    assert text_to_token_str(bytes) == "ãģĵãĤĵãģ«ãģ¡ãģ¯"


def test_text_to_token_str() -> None:
    """Test the conversion from text to token string."""
    normalizer = ByteLevel()
    decoder = ByteLevelDecoder()
    normalized = normalizer.normalize_str("こんにちは")
    byte_form = token_to_bytes(normalized)
    assert byte_form == b"\xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf"
    assert text_to_token_str(byte_form.decode("utf-8")) == normalized

    assert decoder.decode([byte_form.decode("utf-8")]) == "こんにちは"
