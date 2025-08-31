from pydantic import PrivateAttr, RootModel


class VocabMixin:
    """Mixin class for vocabulary-related functionality."""

    @property
    def vocabulary(self) -> dict[str, int]:
        """Returns the vocabulary mapping."""
        raise NotImplementedError()  # pragma: no cover

    def __contains__(self, token: str) -> bool:
        """Checks if a token is in the vocabulary."""
        return token in self.vocabulary

    def __getitem__(self, token: str) -> int:
        """Gets the ID of a token in the vocabulary."""
        return self.vocabulary[token]

    def __len__(self) -> int:
        """Gets the number of tokens in the vocabulary."""
        return len(self.vocabulary)


class Vocabulary(RootModel[dict[str, int]], VocabMixin):
    """A vocabulary mapping tokens to their IDs."""

    @property
    def vocabulary(self) -> dict[str, int]:
        """Returns the vocabulary mapping."""
        return self.root

    @property
    def sorted_vocabulary(self) -> list[str]:
        """Returns the vocabulary mapping sorted by token."""
        return [x[0] for x in sorted(self.root.items(), key=lambda x: x[1])]

    def add_token(self, token: str) -> None:
        """Adds a token to the vocabulary."""
        if token in self.root:
            raise ValueError(f"Token '{token}' already exists in vocabulary.")
        self.root[token] = len(self.root)

    def replace_token(self, old_token: str, new_token: str) -> None:
        """Replaces tokens."""
        if old_token not in self.root:
            raise ValueError(f"Token '{old_token}' does not exist in vocabulary.")
        if new_token in self.root:
            raise ValueError(f"Token '{new_token}' already exists in vocabulary.")
        idx = self.root.pop(old_token)
        self.root[new_token] = idx

    def remove_token(self, token: str) -> None:
        """Removes tokens from the vocabulary."""
        if token not in self.root:
            raise ValueError(f"Token '{token}' does not exist in vocabulary.")
        self.root.pop(token)
        # Rebuild the vocabulary to ensure indices are contiguous
        sorted_tokens, _ = zip(*sorted(self.root.items(), key=lambda x: x[1]))
        self.root = {token: idx for idx, token in enumerate(sorted_tokens)}

    def replace_vocabulary(self, vocabulary: list[str]) -> None:
        """Completely replaces the vocabulary by a vocabulary of the same length."""
        self.root = {token: idx for idx, token in enumerate(vocabulary)}


class UnigramVocabulary(RootModel[list[tuple[str, float]]], VocabMixin):
    """A unigram vocabulary mapping tokens to their scores."""

    _vocabulary: dict[str, int] = PrivateAttr(default_factory=dict, init=False)
    _min_score: float = PrivateAttr(default=-100.0, init=False)

    @property
    def vocabulary(self) -> dict[str, int]:
        """Returns the vocabulary mapping."""
        return self._vocabulary

    @property
    def sorted_vocabulary(self) -> list[str]:
        """Returns the vocabulary mapping sorted by token."""
        return [x[0] for x in sorted(self._vocabulary.items(), key=lambda x: x[1])]

    def model_post_init(self, __context: dict) -> None:
        """Initializes the vocabulary."""
        tokens, scores = zip(*self.root) if self.root else ([], [])
        self._vocabulary = {token: idx for idx, token in enumerate(tokens)}
        self._min_score = min(scores, default=-100.0)

    def add_token(self, token: str) -> None:
        """Adds a token to the vocabulary."""
        if token in self._vocabulary:
            raise ValueError(f"Token '{token}' already exists in vocabulary.")
        self.root.append((token, self._min_score))
        self._vocabulary[token] = len(self.root) - 1

    def replace_token(self, old_token: str, new_token: str) -> None:
        """Removes a token from the vocabulary."""
        if old_token not in self._vocabulary:
            raise ValueError(f"Token '{old_token}' does not exist in vocabulary.")
        if new_token in self._vocabulary:
            raise ValueError(f"Token '{new_token}' already exists in vocabulary.")
        idx = self._vocabulary.pop(old_token)
        self._vocabulary[new_token] = idx
        self.root[idx] = (new_token, self.root[idx][1])

    def remove_token(self, token: str) -> None:
        """Removes a token from the vocabulary."""
        if token not in self._vocabulary:
            raise ValueError(f"Token '{token}' does not exist in vocabulary.")
        idx = self._vocabulary.pop(token)
        sorted_tokens, _ = zip(*sorted(self._vocabulary.items(), key=lambda x: x[1]))
        self._vocabulary = {token: idx for idx, token in enumerate(sorted_tokens)}

        self.root.pop(idx)

    def replace_vocabulary(self, vocabulary: list[str]) -> None:
        """Completely replaces the vocabulary by a vocabulary of the same length."""
        self.root = [(token, self.root[idx][1]) for idx, token in enumerate(vocabulary)]
        self._vocabulary = {token: idx for idx, token in enumerate(vocabulary)}
