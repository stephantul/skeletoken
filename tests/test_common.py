import re

import pytest
import regex

from skeletoken.common import coerce_string_regex_pattern


def test_coercion_to_string_regex() -> None:
    """Tests the coercion function for string and regex patterns."""
    assert coerce_string_regex_pattern("test") == {"String": "test"}
    assert coerce_string_regex_pattern(re.compile("test")) == {"Regex": "test"}
    assert coerce_string_regex_pattern(regex.compile("test")) == {"Regex": "test"}

    with pytest.raises(TypeError):
        assert coerce_string_regex_pattern(123)  # type: ignore  # On purpose
