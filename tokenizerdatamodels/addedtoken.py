from pydantic import BaseModel


class AddedToken(BaseModel):
    content: str
    single_word: bool
    lstrip: bool
    rstrip: bool
    normalized: bool
    special: bool
    id: int
