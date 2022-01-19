"""Tests for the decision tree modularization"""
import pytest

from tedana.selection import ComponentSelector

# ----------------------------------------------------------------------
# ComponentSelector Tests
# ----------------------------------------------------------------------

# load_config
# -----------
def test_load_config_fails():
    """Tests for load_config failure modes"""

    # We recast to ValueError in the file not found and directory cases
    with pytest.raises(ValueError):
        ComponentSelector.load_config("THIS FILE DOES NOT EXIST.txt")

    # Raises IsADirectoryError for a directory
    with pytest.raises(ValueError):
        ComponentSelector.load_config(".")

    # Note: we defer validation errors for validate_tree even though
    # load_config may raise them


def test_load_config_succeeds():
    """Tests to make sure load_config succeeds"""

    # The minimal tree should have an id of "minimal_decision_tree_test1"
    tree = ComponentSelector.load_config("minimal")
    assert tree["tree_id"] == "minimal_decision_tree_test1"


# validate_tree
# -------------
def test_validate_tree_fails():
    """Tests to make sure validate_tree fails for invalid trees"""

    # An empty dict should not be valid
    with pytest.raises(ComponentSelector.TreeError):
        ComponentSelector.validate_tree({})
