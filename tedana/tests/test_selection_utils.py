"""Tests for the tedana.selection.selection_utils module."""
import numpy as np
import pytest

from tedana.selection import selection_utils


def test_getelbow_smoke():
    """A smoke test for the getelbow function."""
    arr = np.random.random(100)
    idx = selection_utils.getelbow(arr)
    assert isinstance(idx, np.integer)

    val = selection_utils.getelbow(arr, return_val=True)
    assert isinstance(val, float)

    # Running an empty array should raise a ValueError
    arr = np.array([])
    with pytest.raises(ValueError):
        selection_utils.getelbow(arr)

    # Running a 2D array should raise a ValueError
    arr = np.random.random((100, 100))
    with pytest.raises(ValueError):
        selection_utils.getelbow(arr)


def test_getelbow_cons():
    """A smoke test for the getelbow_cons function."""
    arr = np.random.random(100)
    idx = selection_utils.getelbow_cons(arr)
    assert isinstance(idx, np.integer)

    val = selection_utils.getelbow_cons(arr, return_val=True)
    assert isinstance(val, float)

    # Running an empty array should raise a ValueError
    arr = np.array([])
    with pytest.raises(ValueError):
        selection_utils.getelbow_cons(arr)

    # Running a 2D array should raise a ValueError
    arr = np.random.random((100, 100))
    with pytest.raises(ValueError):
        selection_utils.getelbow_cons(arr)
