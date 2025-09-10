from pydantic import BaseModel, RootModel


class AddedToken(BaseModel):
    """
    Represents an added token in a tokenizer.

    An added token can be a special token or a regular token that is not necessarily part of the original vocabulary.
    Note that AddedToken can be used to represent multiword units. For example, a token like
    "New York" can be represented as a single AddedToken with content "New York".

    Multiword units are only supported if single_word is set to False.

    Attributes
    ----------
    content : str
        The string content of the token.
    single_word : bool
        Indicates if the token is a single word.
        If it is not a single word, it may be a subword or a character.
        e.g., if single_word is False, the token could be a subword like "ing".
    lstrip : bool
        If set, the token will be stripped from the left side.
    rstrip : bool
        If set, the token will be stripped from the right side.
    normalized : bool
        If set, the token form is searched _before_ normalization and pretokenization happens.
        For example, if you use a lowercase normalizer, the token "Hello" will be found as "hello",
        but only if normalized is set to True. If it is False, "Hello" will be found as is.
    special : bool
        If set, the token is a special token.
        Special tokens are skipped during decoding, and are represented as single tokens in the vocabulary.
        Special tokens are typically used to represent specific concepts or actions,
        such as [CLS] for classification or [SEP] for separation.
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

    def __getitem__(self, index: int) -> AddedToken:
        """Get an addedtoken by index."""
        return self.root[index]

    def get_token(self, token: str) -> AddedToken | None:
        """Returns the added token for a given form."""
        for added_token in self.root:
            if added_token.content == token:
                return added_token
        return None

    def maybe_add_token(
        self,
        token: str,
        id: int,
        is_special: bool = False,
        normalized: bool = False,
        single_word: bool = True,
        lstrip: bool = True,
        rstrip: bool = True,
    ) -> None:
        """Adds a new added token."""
        # If the token already exists, update it. Don't create a new one.
        if added_token := self.get_token(token):
            added_token.special = is_special
            added_token.normalized = normalized
            added_token.single_word = single_word
            added_token.lstrip = lstrip
            added_token.rstrip = rstrip
            added_token.id = id
        else:
            new_token = AddedToken(
                content=token,
                special=is_special,
                normalized=normalized,
                single_word=single_word,
                lstrip=lstrip,
                rstrip=rstrip,
                id=id,
            )
            self.root.append(new_token)

    def maybe_remove_token(self, token: str) -> None:
        """Removes the added token for a given form if it exists."""
        self.root = [added_token for added_token in self.root if added_token.content != token]

    def maybe_replace_token(self, old_token: str, new_token: str) -> None:
        """Replaces the added token for a given form, if it exists."""
        added_token = self.get_token(old_token)
        if added_token:
            added_token.content = new_token

    def get_special_tokens(self) -> list[AddedToken]:
        """Returns a list of all special added tokens."""
        return [token for token in self.root if token.special]

    def get_unnormalized_tokens(self) -> list[AddedToken]:
        """Returns a list of all unnormalized added tokens."""
        return [token for token in self.root if not token.normalized]

    def __len__(self) -> int:
        """Returns the number of added tokens."""
        return len(self.root)
