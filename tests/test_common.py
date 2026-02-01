import re

import pytest
import regex

from skeletoken.common import RegexPattern, StringPattern, coerce_string_regex_pattern


def test_coercion_to_string_regex() -> None:
    """Tests the coercion function for string and regex patterns."""
    expected_string = StringPattern.model_validate({"String": "test"})
    expected_regex = RegexPattern.model_validate({"Regex": "test"})
    assert coerce_string_regex_pattern("test") == expected_string
    assert coerce_string_regex_pattern(re.compile("test")) == expected_regex
    assert coerce_string_regex_pattern(regex.compile("test")) == expected_regex

    with pytest.raises(TypeError):
        assert coerce_string_regex_pattern(123)  # type: ignore  # On purpose
