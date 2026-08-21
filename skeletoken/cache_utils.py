from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar, cast

if TYPE_CHECKING:
    from skeletoken.base import TokenizerModel  # pragma: nocover

F = TypeVar("F", bound=Callable[..., Any])


def resets_tokenizer_cache(method: F) -> F:
    """Reset the cached tokenizer after the wrapped method runs."""

    @wraps(method)
    def wrapper(self: TokenizerModel, *args: Any, **kwargs: Any) -> Any:
        result = method(self, *args, **kwargs)
        self._tokenizer = None
        return result

    return cast(F, wrapper)


def resets_preprocessor_cache(method: F) -> F:
    """Reset the cached preprocessor and tokenizer after the wrapped method runs."""

    @wraps(method)
    def wrapper(self: TokenizerModel, *args: Any, **kwargs: Any) -> Any:
        result = method(self, *args, **kwargs)
        self._preprocessor = None
        self._tokenizer = None
        return result

    return cast(F, wrapper)
