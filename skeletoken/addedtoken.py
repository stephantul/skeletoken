from pydantic import BaseModel


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
