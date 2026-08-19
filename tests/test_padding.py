from skeletoken.padding import BatchLongestStrategy, FixedStrategy, Padding, is_basic_padding


def test_is_basic_padding_none() -> None:
    """None padding is never basic."""
    assert not is_basic_padding(None)


def test_is_basic_padding_fixed_zero() -> None:
    """A Fixed strategy with length 0 is the basic padding hack."""
    padding = Padding(strategy=FixedStrategy(Fixed=0), pad_id=0, pad_type_id=0, pad_token="[PAD]")
    assert is_basic_padding(padding)


def test_is_basic_padding_fixed_nonzero() -> None:
    """A Fixed strategy with a nonzero length is real padding, not basic."""
    padding = Padding(strategy=FixedStrategy(Fixed=8), pad_id=0, pad_type_id=0, pad_token="[PAD]")
    assert not is_basic_padding(padding)


def test_is_basic_padding_batch_longest() -> None:
    """A BatchLongest strategy is real padding, not basic."""
    padding = Padding(strategy=BatchLongestStrategy("BatchLongest"), pad_id=0, pad_type_id=0, pad_token="[PAD]")
    assert not is_basic_padding(padding)


def test_is_basic_padding_default_strategy() -> None:
    """The default strategy on Padding is the basic Fixed(0) hack."""
    padding = Padding(pad_id=0, pad_type_id=0, pad_token="[PAD]")
    assert is_basic_padding(padding)
