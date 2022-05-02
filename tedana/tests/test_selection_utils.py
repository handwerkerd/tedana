"""Tests for the tedana.selection.selection_utils module."""
import numpy as np
import pytest
import os
import pandas as pd

from tedana.selection.ComponentSelector import ComponentSelector
from tedana.selection import selection_utils


THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def sample_selector():
    """
    Retrieves a sample component table and initializes
    a selector using that component table and the minimal tree
    """

    tree = "minimal"

    sample_fname = os.path.join(THIS_DIR, "data", "sample_comptable.tsv")
    component_table = pd.read_csv(sample_fname, delimiter="\t")
    component_table["classification_tags"] = ""

    xcomp = {
        "n_echos": 3,
        "n_vols": 201,
    }

    return ComponentSelector(tree, component_table, cross_component_metrics=xcomp)


def test_selectcomps2use_succeeds():
    """
    Tests to make sure selectcomps2use runs with full range of inputs.
    Include tests to make sure the correct number of components are selected
    from the pre-defined sample_comptable.tsv component table
    """
    selector = sample_selector()

    decide_comps_options = [
        "rejected",
        ["accepted"],
        "all",
        ["accepted", "rejected"],
        4,
        [2, 6, 4],
        "NotALabel",
    ]
    decide_comps_lengths = [4, 17, 21, 21, 1, 3, None]
    for idx, decide_comps in enumerate(decide_comps_options):
        assert selection_utils.selectcomps2use(
            selector, decide_comps
        ), f"selectcomps2use crashed with decide_comps={decide_comps}"
        comps2use, component_table = selection_utils.selectcomps2use(selector, decide_comps)
        if decide_comps_lengths[idx]:
            assert (
                len(comps2use) > 0
            ), f"selectcomps2use test should select {decide_comps_lengths[idx]} with decide_comps={decide_comps}, but it selected {len(comps2use)}"
        else:
            assert (
                comps2use == None
            ), f"selectcomps2use test should output None with decide_comps={decide_comps}, but it selected {len(comps2use)}"


def test_selectcomps2use_fails():
    """Tests for selectcomps2use failure modes"""
    selector = sample_selector()

    decide_comps_options = [
        18.2,  # no floats
        [11.2, 13.1],  # no list of floats
        ["accepted", 4],  # needs to be either int or string, not both
        [4, 3, -1, 9],  # no index should be < 0
        [2, 4, 6, 21],  # no index should be > number of 0 indexed components
        22,  ## no index should be > number of 0 indexed components
    ]
    for decide_comps in decide_comps_options:
        with pytest.raises(ValueError):
            selection_utils.selectcomps2use(selector, decide_comps)


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
