import logging
from typing import Any

from pydantic import PrivateAttr, RootModel

logger = logging.getLogger(__name__)

_Merge = tuple[str, str]


class Merges(RootModel[list[_Merge]]):
    """Data model representing a list of merge operations."""

    _merge_index: dict[_Merge, int] = PrivateAttr(default_factory=dict)
    _all_merge_tokens: set[str] = PrivateAttr(default_factory=set)

    def model_post_init(self, context: Any) -> None:
        """Post init of the model."""
        self._merge_index = {merge: i for i, merge in enumerate(self.root)}
        self._all_merge_tokens = set()
        for left, right in self.root:
            self._all_merge_tokens.add(left)
            self._all_merge_tokens.add(right)

    def _add_merges_for_token(self, token: str) -> None:
        """Add merge operations for a specific token."""
        # Convert to tuple for typing.
        token_form = tuple(self._merge(token))
        while len(token_form) > 1:
            for index in range(0, len(token_form) - 1, 2):
                left, right = token_form[index], token_form[index + 1]
                bigram = (left, right)
                self.root.append(bigram)
                self._merge_index[bigram] = len(self.root) - 1
                self._all_merge_tokens.add(left)
                self._all_merge_tokens.add(right)
            token_form = tuple(self._merge(token))

    def _merge(self, token: str) -> list[str]:
        """Merge a token with its subword components."""
        chars = list(token)

        # Special case, the token is length 1.
        if len(chars) == 1:
            return chars

        while True:
            string_bigrams = _bigrams(chars)
            lowest_idx = float("inf")
            merge: _Merge = ("", "")
            for bigram in string_bigrams:
                if bigram not in self._merge_index:
                    continue
                idx = self._merge_index[bigram]
                if idx < lowest_idx:
                    lowest_idx = idx
                    merge = bigram
            if lowest_idx == float("inf"):
                break
            chars = _merge(chars, merge)

        return chars


def _bigrams(chars: list[str]) -> list[_Merge]:
    """Calculate the bigrams of a string."""
    return list(zip(chars[:-1], chars[1:]))


def _merge(token: list[str], bigram: _Merge) -> list[str]:
    """Merge a bytegram into a token."""
    # Merge the bytegram into the token
    new_token = []
    index = 0
    while index < len(token) - 1:
        if (token[index], token[index + 1]) == bigram:
            new_token.append("".join(bigram))
            index += 1
        else:
            new_token.append(token[index])
        index += 1
    if index < len(token):
        new_token.append(token[index])
    return new_token
