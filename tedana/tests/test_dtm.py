"""Tests for the decision tree modularization"""
import pytest
import json, os, glob

from tedana.selection import ComponentSelector

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

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


def test_validate_tree_succeeds():
    """
    Tests to make sure validate_tree suceeds for valid decision trees
    Tested on all default trees in ../resources/decision_trees
    Note: If there is a tree in the default trees directory that
    is being developed and not yet valid, it's file name should
    include 'invalid' as a prefix
    Also checks trees in data/ComponentSelection that have the
    'valid_trees' prefix. These are for trees that aren't to be
    used, but are designed tomake sure valid edge cases do work
    """

    # Get the names of all trees that are included as default options
    default_tree_names = glob.glob(
        os.path.join(THIS_DIR, "../resources/decision_trees/[!invalid]*.json")
    )
    # Get the names of trees that are used to test edge cases for a valid tree
    test_tree_names = glob.glob(
        os.path.join(THIS_DIR, "data/ComponentSelection/", "valid_trees_*.json")
    )
    tree_names = default_tree_names + test_tree_names
    for tree_name in tree_names:
        f = open(tree_name)
        tree = json.load(f)
        print(f"Validating: {tree_name}")
        assert ComponentSelector.validate_tree(tree)


def test_validate_tree_fails():
    """
    Tests to make sure validate_tree fails for invalid trees
    Tests ../resources/decision_trees/invalid*.json and
    ./data/ComponentSelection/invalid*.json trees
    """

    # An empty dict should not be valid
    with pytest.raises(ComponentSelector.TreeError):
        ComponentSelector.validate_tree({})

    # Get the names of all trees that are included as default options with the
    # invalid prefix
    default_tree_names = glob.glob(
        os.path.join(THIS_DIR, "../resources/decision_trees/invalid*.json")
    )
    # Get the names of trees that are used to test edge cases for an invalid tree
    test_tree_names = glob.glob(
        os.path.join(THIS_DIR, "data/ComponentSelection/", "invalid_trees_*.json")
    )
    tree_names = default_tree_names + test_tree_names
    for tree_name in tree_names:
        f = open(tree_name)
        tree = json.load(f)
        print(f"Validating: {tree_name}")
        with pytest.raises(ComponentSelector.TreeError):
            ComponentSelector.validate_tree(tree)
