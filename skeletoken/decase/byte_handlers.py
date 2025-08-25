"""Utilities to handle byte conversion in decasing."""


def bytes_to_unicode() -> dict[int, str]:
    """Converts byte values to Unicode characters for byte-level tokenization."""
    input_ids = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    output_ids = input_ids[:]
    n = 0
    for char_ord in range(256):
        if char_ord not in input_ids:
            input_ids.append(char_ord)
            output_ids.append(256 + n)
            n += 1
    output_chars = [chr(c) for c in output_ids]
    return dict(zip(input_ids, output_chars, strict=True))


_MAPPING = bytes_to_unicode()
_INV_MAPPING = {v: k for k, v in _MAPPING.items()}


def token_to_bytes(token: str) -> bytes:
    """Convert a token string to bytes using the byte-level mapping."""
    # Each character in a byte-level token maps to exactly one byte.
    # We need to use .get because some special characters may not be in the mapping.
    # This will return 0 for any character not in the mapping.
    try:
        return bytes(_INV_MAPPING[char] for char in token)
    except KeyError as e:
        raise ValueError(f"Token contains characters that cannot be mapped to bytes: {token}") from e


def text_to_token_str(text: str) -> str:
    """Convert a text string to a token string using the byte-level mapping."""
    # Encode text as UTF-8 bytes, then map each byte back to the placeholder char.
    encoded = text.encode("utf-8")
    return "".join(_MAPPING[byte] for byte in encoded)
