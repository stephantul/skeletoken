import pytest

from skeletoken.merges import Merges


def test_merges(small_merges: Merges) -> None:
    """Test simple merge functionality."""
    assert small_merges.root == [("a", "b"), ("c", "d"), ("ab", "c"), ("a", "bc")]
    assert small_merges._merge_index == {
        ("a", "b"): 0,
        ("c", "d"): 1,
        ("ab", "c"): 2,
        ("a", "bc"): 3,
    }


def test_merge(small_merges: Merges) -> None:
    """Test the merge function."""
    assert small_merges._merge("abc") == ["abc"]
    # Single character just returns
    assert small_merges._merge("a") == ["a"]
    assert small_merges._merge("abcd") == ["ab", "cd"]
    assert small_merges._merge("bca") == ["b", "c", "a"]
    # Nothing in vocab just returns all single characters
    assert small_merges._merge("xyz") == ["x", "y", "z"]


@pytest.mark.parametrize(
    "token, before, added",
    [
        ("ab", ["ab"], [("a", "b"), ("c", "d"), ("ab", "c"), ("a", "bc")]),
        ("cd", ["cd"], [("a", "b"), ("c", "d"), ("ab", "c"), ("a", "bc")]),
        ("abc", ["abc"], [("a", "b"), ("c", "d"), ("ab", "c"), ("a", "bc")]),
        ("bc", ["b", "c"], [("a", "b"), ("c", "d"), ("ab", "c"), ("a", "bc"), ("b", "c")]),
        (
            "elephantine",
            ["e", "l", "e", "p", "h", "a", "n", "t", "i", "n", "e"],
            [
                ("a", "b"),
                ("c", "d"),
                ("ab", "c"),
                ("a", "bc"),
                ("e", "l"),
                ("e", "p"),
                ("h", "a"),
                ("n", "t"),
                ("i", "n"),
                ("el", "ep"),
                ("ha", "nt"),
                ("in", "e"),
                ("elep", "hant"),
                ("elephant", "ine"),
            ],
        ),
    ],
)
def test_add_merges(small_merges: Merges, token: str, before: list[str], added: list[str]) -> None:
    """Test whether merges are added."""
    assert small_merges._merge(token) == before
    small_merges._add_merges_for_token(token)
    assert small_merges._merge(token) == [token]
    assert small_merges.root == added
