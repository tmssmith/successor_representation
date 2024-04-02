# pylint: disable=missing-function-docstring


import numpy as np
import pytest
from successor_representation.utils import make_hashable


def is_hashable(x):
    try:
        hash(x)
        return True
    except Exception:
        return False


def test_make_hashable():
    assert is_hashable(make_hashable(1)) is True
    assert is_hashable(make_hashable("test")) is True
    assert is_hashable(make_hashable(np.array([1, 2, 3, 4]))) is True
    assert is_hashable(make_hashable(np.array([[1, 2, 3], [4, 5, 6]]))) is True
    assert (
        is_hashable(
            make_hashable(np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]]))
        )
        is True
    )
    assert is_hashable(make_hashable([1, 2, 3])) is True
    print(make_hashable([[1, 2, 3], [4, 5, 6]]))
    assert is_hashable(make_hashable([[1, 2, 3], [4, 5, 6]])) is True
    assert (
        is_hashable(make_hashable([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]]))
        is True
    )

    with pytest.raises(NotImplementedError):
        make_hashable(None)
    with pytest.raises(NotImplementedError):
        make_hashable({})
    with pytest.raises(NotImplementedError):
        make_hashable(set([0, 1, 2, 3]))
