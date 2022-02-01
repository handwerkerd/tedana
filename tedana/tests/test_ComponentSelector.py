"""Tests for the decision tree modularization"""
import pytest
import pandas as pd
import json, os, glob

from tedana.selection import ComponentSelector

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------
# Functions Used For Tests
# ----------------------------------------------------------------------


def dicts_to_test(treechoice):
    """
    Outputs decision tree dictionaries to use to test tree validation

    Parameters
    ----------
    treechoice: :obj:`str` One of several labels to select which dict to output
        Options are:
        "valid": A valid tree without any warnings
        "warnings": A tree that would trigger all warnings, but pass validation
        "extra_req_param": A tree with an undefined required parameter for a decision node function
        "extra_opt_param": A tree with an undefined optional parameter for a decision node function
        "missing_req_param": A missing required param in a decision node function
        "missing_function": An undefined decision node function
        "missing_key": A dict missing one of the required keys (refs)

    Returns
    -------
    tree: :ojb:`dict` A dict that can be input into ComponentSelector.validate_tree
    """

    # valid_dict is a simple valid dictionary to test
    valid_dict = {
        "tree_id": "valid_simple_tree",
        "info": "This is a short valid tree",
        "report": "",
        "refs": "",
        "necessary_metrics": ["kappa", "rho"],
        "intermediate_classifications": ["random1", "random2"],
        "classification_tags": ["Random1", "Random2"],
        "nodes": [
            {
                "functionname": "dec_left_op_right",
                "parameters": {
                    "ifTrue": "rejected",
                    "ifFalse": "nochange",
                    "decide_comps": "all",
                    "op": ">",
                    "left": "rho",
                    "right": "kappa",
                },
                "kwargs": {
                    "log_extra_info": "random1 if Kappa<Rho",
                    "tag_ifTrue": "random1",
                },
            },
            {
                "functionname": "dec_left_op_right",
                "parameters": {
                    "ifTrue": "random2",
                    "ifFalse": "nochange",
                    "decide_comps": "all",
                    "op": ">",
                    "left": "kappa",
                    "right": "rho",
                },
                "kwargs": {
                    "log_extra_info": "random2 if Kappa>Rho",
                    "log_extra_report": "",
                    "tag_ifTrue": "random2",
                },
            },
            {
                "functionname": "manual_classify",
                "parameters": {
                    "new_classification": "accepted",
                    "decide_comps": "random2",
                },
                "kwargs": {
                    "log_extra_info": "",
                    "log_extra_report": "",
                    "tag": "Random2",
                },
            },
            {
                "functionname": "manual_classify",
                "parameters": {
                    "new_classification": "rejected",
                    "decide_comps": "random1",
                },
                "kwargs": {
                    "tag": "Random1",
                },
            },
        ],
    }

    tree = valid_dict
    if treechoice == "valid":
        return tree
    elif treechoice == "warnings":
        tree["unused_key"] = "There can be added keys that are valid, but are not used"
        tree["nodes"][1]["kwargs"]["tag_ifTrue"] = "classification_not_predefined"
        tree["nodes"][2]["parameters"]["decide_comps"] = "classification_not_predefined"
        tree["nodes"][2]["kwargs"]["tag"] = "tag_not_predefined"

    elif treechoice == "extra_req_param":
        tree["nodes"][0]["parameters"]["nonexistent_req_param"] = True
    elif treechoice == "extra_opt_param":
        tree["nodes"][0]["kwargs"]["nonexistent_opt_param"] = True
    elif treechoice == "missing_req_param":
        tree["nodes"][0]["parameters"].pop("op")
    elif treechoice == "missing_function":
        tree["nodes"][0]["functionname"] = "not_a_function"
    elif treechoice == "missing_key":
        tree.pop("refs")
    else:
        raise Exception(f"{treechoice} is an invalid option for treechoice")

    return tree


def component_table_to_test(tablechoice):
    """
    A default component table to use for various tests

    Parameters
    ----------
    tablechoice: :obj:`str`
        One of several labels to select which component_table to output
        Options are:
        "valid": A valid table without any warnings
        "notDF": A variable that is an int rather than a dataframe
        "noComponent": A table missing the "Component" column
    """

    comp_dict = {
        "Component": [0, 1, 2, 3, 4, 5, 6],
        "kappa": [100, 90, 80, 70, 60, 50, 40],
        "rho": [40, 50, 60, 70, 80, 90, 100],
        "dice_FT2": [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7],
        "dice_FS0": [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
        "countsigFT2": [5000, 5000, 5000, 5000, 5000, 5000, 5000],
        "countsigFS0": [4000, 4000, 4000, 4000, 4000, 4000, 4000],
        "variance explained": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4],
        "signal-noise_t": [3000, 3000, 3000, 3000, 3000, 3000, 3000],
    }

    component_table = pd.DataFrame(data=comp_dict)
    if tablechoice == "valid":
        return component_table
    elif tablechoice == "notDF":
        return 5
    elif tablechoice == "noComponent":
        component_table.drop("Component")
    else:
        raise Exception(f"{tablechoice} is an invalid option for tablechoice")

    return component_table


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
    Tests to make sure validate_tree suceeds for all default
    decision trees in  decision trees
    Tested on all default trees in ./tedana/resources/decision_trees
    Note: If there is a tree in the default trees directory that
    is being developed and not yet valid, it's file name should
    include 'invalid' as a prefix
    """

    default_tree_names = glob.glob(
        os.path.join(THIS_DIR, "../resources/decision_trees/[!invalid]*.json")
    )

    for tree_name in default_tree_names:
        f = open(tree_name)
        tree = json.load(f)
        assert ComponentSelector.validate_tree(tree)


def test_validate_tree_warnings():
    """
    Tests to make sure validate_tree triggers all warning conditions
    but still succeeds
    """

    # A tree that raises all possible warnings in the validator should still be valid
    assert ComponentSelector.validate_tree(dicts_to_test("warnings"))


def test_validate_tree_fails():
    """
    Tests to make sure validate_tree fails for invalid trees
    Tests ../resources/decision_trees/invalid*.json and
    ./data/ComponentSelection/invalid*.json trees
    """

    # An empty dict should not be valid
    with pytest.raises(ComponentSelector.TreeError):
        ComponentSelector.validate_tree({})

    # A tree that is missing a required key should not be valid
    with pytest.raises(ComponentSelector.TreeError):
        ComponentSelector.validate_tree(dicts_to_test("missing_key"))

    # Calling a selection node function that does not exist should not be valid
    with pytest.raises(ComponentSelector.TreeError):
        ComponentSelector.validate_tree(dicts_to_test("missing_function"))

    # Calling a function with an non-existent required parameter should not be valid
    with pytest.raises(ComponentSelector.TreeError):
        ComponentSelector.validate_tree(dicts_to_test("extra_req_param"))

    # Calling a function with an non-existent optional parameter should not be valid
    with pytest.raises(ComponentSelector.TreeError):
        ComponentSelector.validate_tree(dicts_to_test("extra_opt_param"))

    # Calling a function missing a required parameter should not be valid
    with pytest.raises(ComponentSelector.TreeError):
        ComponentSelector.validate_tree(dicts_to_test("missing_req_param"))


# validate_component_table
# -------------


def test_validate_component_table_succeeds():
    """
    Tests if validate component table succeeds
    """

    assert ComponentSelector.validate_component_table(component_table_to_test("valid"))


def test_validate_component_table_fails():
    """
    Tests if validate component table fails
    """
    with pytest.raises(TypeError):
        ComponentSelector.validate_component_table(component_table_to_test("notDF"))

    with pytest.raises(KeyError):
        ComponentSelector.validate_component_table(
            component_table_to_test("noComponent")
        )


# ComponenentSelector __init__
# ----------------------------


def test_Selector_init_succeeds():
    """
    Tests if Selector class initialization succeeds

    TODO: Right now there is no failure checks within initialization
    so there is no reason to make a failure test
    The tree is validated in a sub-fuction and there is another
    function that can be used to validate the component table.
    Should anything else be validated directly within this
    initialization?
    """

    assert ComponentSelector.ComponentSelector(
        "minimal",
        component_table_to_test("valid"),
        n_echos=3,
        n_vols=200,
        irrelivant="arbitrary inputs are allowed",
    )


# ComponenentSelector select
# ----------------------------


def test_Selector_select_succeeds():
    """
    Tests if ComponentSelector.select succeeds

    TODO: Right now failure checks are in subfunctions, not
    this function, so there is no reason to make a failure test
    Should anything else be validated directly within this
    function?
    """

    selector = ComponentSelector.ComponentSelector(
        "minimal", component_table_to_test("valid"), n_echos=3
    )
    selector.select()


# TODO It's hard to test check_null when it's not actually being used in the minimal tree
#   Testing for this function should be added when the kundu tree is revived.
# def test_check_null():
#    params = self.check_null(params, node["functionname"])
#             kwargs = self.check_null(kwargs, node["functionname"])


def test_are_only_necessary_metrics_used_succeeds():
    """
    Test that are_only_necessary_metrics_used runs with no warnings
    and with every condition that would trigger a warning
    TODO: I can't used assert when a function doesn't return
    anything so I just try to run this then?
    """

    selector = ComponentSelector.ComponentSelector(
        "minimal", component_table_to_test("valid")
    )

    # Warning: necessary metrics aren't used
    selector.used_metrics = set()
    selector.are_only_necessary_metrics_used()

    # No warning: necessary_metrics == used_metrics
    selector.used_metrics = selector.necessary_metrics
    selector.are_only_necessary_metrics_used()

    # Warning: used_metrics includes a metric not listed as necessary
    selector.used_metrics.add("undeclared")
    selector.are_only_necessary_metrics_used()


def test_are_all_components_accepted_or_rejected_succeeds():
    """
    tests are_all_components_accepted_or_rejected runs with
    and without triggering warnings
    """

    selector = ComponentSelector.ComponentSelector(
        "minimal", component_table_to_test("valid")
    )

    selector.component_table["classification"] = "accepted"

    # Runs without warnings
    selector.are_all_components_accepted_or_rejected()

    # Runs with a warning that non-final classificaitons remsin
    selector.component_table["classification"][0] = "Intermediate1"
    selector.component_table["classification"][1] = "Intermediate2"
    selector.component_table["classification"][2] = "Intermediate2"
    selector.are_all_components_accepted_or_rejected()
