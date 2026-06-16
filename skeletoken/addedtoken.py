from typing import Any

from pydantic import BaseModel, PrivateAttr, RootModel


class AddedToken(BaseModel):
    """Represents an added token in a tokenizer.

    An added token can be a special token or a regular token that is not necessarily part of the original vocabulary.
    Note that AddedToken can be used to represent multiword units. For example, a token like
    "New York" can be represented as a single AddedToken with content "New York".

    Multiword units are only supported if single_word is set to False.

    Attributes
    ----------
    content : str
        The string content of the token.
    single_word : bool
        Indicates if the token is a single word. If it is not a single word, it may be a subword or a character.
        e.g., if single_word is False, the token could be a subword like "ing".
    lstrip : bool
        If set, the token will be stripped from the left side.
    rstrip : bool
        If set, the token will be stripped from the right side.
    normalized : bool
        If set, the token is already in its normalized form and will be kept as-is during
        vocabulary consolidation. If False, the token will be re-preprocessed when the
        tokenizer's normalization or pretokenization rules change.
    special : bool
        If set, the token is a special token.
        Special tokens are skipped during decoding, and are represented as single tokens in the vocabulary.
        Special tokens are typically used to represent specific concepts or actions, such as [CLS] for classification
        or [SEP] for separation.
    id : int
        The unique identifier for the token.

    """

    content: str
    single_word: bool
    lstrip: bool
    rstrip: bool
    normalized: bool
    special: bool
    id: int


class AddedTokens(RootModel[list[AddedToken]]):
    """Represents a collection of AddedTokens."""

    _index: dict[str, AddedToken] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """Build the content→token index."""
        self._index = {t.content: t for t in self.root}

    def __getitem__(self, index: int) -> AddedToken:
        """Get an addedtoken by index."""
        return self.root[index]

    def get_token(self, token: str) -> AddedToken | None:
        """Return the added token for a given form."""
        return self._index.get(token)

    def upsert_token(
        self,
        token: str,
        id: int,
        is_special: bool | None = None,
        normalized: bool | None = None,
        single_word: bool | None = None,
        lstrip: bool | None = None,
        rstrip: bool | None = None,
    ) -> None:
        """Add a token, or update its id in-place if it already exists.

        When the token already exists, only `id` is updated; omitted keyword
        arguments leave the existing field values unchanged.  Pass an explicit
        value to also update that field on an existing token.
        """
        if added_token := self._index.get(token):
            added_token.id = id
            if is_special is not None:
                added_token.special = is_special
            if normalized is not None:
                added_token.normalized = normalized
            if single_word is not None:
                added_token.single_word = single_word
            if lstrip is not None:
                added_token.lstrip = lstrip
            if rstrip is not None:
                added_token.rstrip = rstrip
        else:
            new_token = AddedToken(
                content=token,
                special=is_special or False,
                normalized=normalized or False,
                single_word=single_word if single_word is not None else True,
                lstrip=lstrip if lstrip is not None else True,
                rstrip=rstrip if rstrip is not None else True,
                id=id,
            )
            self.root.append(new_token)
            self._index[token] = new_token

    def maybe_remove_token(self, token: str) -> None:
        """Remove the added token for a given form if it exists."""
        if token in self._index:
            self._index.pop(token)
            self.root = [t for t in self.root if t.content != token]

    def maybe_replace_token(self, old_token: str, new_token: str) -> None:
        """Replace the added token for a given form, if it exists."""
        if old_token == new_token:
            return
        added_token = self._index.get(old_token)
        if added_token:
            self.maybe_remove_token(new_token)
            self._index.pop(old_token)
            added_token.content = new_token
            self._index[new_token] = added_token

    def get_special_tokens(self) -> list[AddedToken]:
        """Return a list of all special added tokens."""
        return [token for token in self.root if token.special]

    def __len__(self) -> int:
        """Return the number of added tokens."""
        return len(self.root)
