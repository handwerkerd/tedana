"""Tests for the tedana.selection.selection_nodes module."""
from re import S
import numpy as np
import pytest
import os
import pandas as pd

from tedana.selection.ComponentSelector import ComponentSelector
from tedana.selection import selection_utils
from tedana.selection import selection_nodes
from tedana.tests.test_selection_utils import sample_component_table, sample_selector

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def test_manual_classify_smoke():
    """Smoke tests for all options in manual_classify"""

    selector = sample_selector(options="provclass")

    decide_comps = "provisional accept"
    new_classification = "accepted"

    # Outputs just the metrics used in this function (nothing in this case)
    used_metrics = selection_nodes.manual_classify(
        selector, decide_comps, new_classification, only_used_metrics=True
    )
    assert used_metrics == set()

    # Standard execution where components are changed from "provisional accept" to "accepted"
    # And all extra logging code is run
    selector = selection_nodes.manual_classify(
        selector,
        decide_comps,
        new_classification,
        log_extra_report="report log",
        log_extra_info="info log",
        custom_node_label="custom label",
        tag="test tag",
    )
    # There should be 4 selected components and component_status_table should have a new column "Node 0"
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numTrue"] == 4
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numFalse"] == 0
    assert f"Node {selector.current_node_idx}" in selector.component_status_table

    # No components with "NotALabel" classification so nothing selected and no
    #   Node 1 column not created in component_status_table
    selector.current_node_idx = 1
    selector = selection_nodes.manual_classify(selector, "NotAClassification", new_classification)
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numTrue"] == 0
    assert f"Node {selector.current_node_idx}" not in selector.component_status_table

    # Changing components from "rejected" to "accepted" and suppressing warning
    selector.current_node_idx = 2
    selector = selection_nodes.manual_classify(
        selector,
        "rejected",
        new_classification,
        clear_classification_tags=True,
        log_extra_report="report log",
        log_extra_info="info log",
        tag="test tag",
        dont_warn_reclassify=True,
    )
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numTrue"] == 4
    assert f"Node {selector.current_node_idx}" in selector.component_status_table


def test_dec_left_op_right_succeeds():
    """tests for successful calls to dec_left_op_right"""

    selector = sample_selector(options="provclass")

    decide_comps = "provisional accept"

    # Outputs just the metrics used in this function {"kappa", "rho"}
    used_metrics = selection_nodes.dec_left_op_right(
        selector, "accepted", "rejected", decide_comps, ">", "kappa", "rho", only_used_metrics=True
    )
    assert len(used_metrics - {"kappa", "rho"}) == 0

    # Standard execution where components with kappa>rho are changed from "provisional accept" to "accepted"
    # And all extra logging code and options are run
    # left and right are both component_table_metrics
    selector = selection_nodes.dec_left_op_right(
        selector,
        "accepted",
        "rejected",
        decide_comps,
        ">",
        "kappa",
        "rho",
        left_scale=0.9,
        right_scale=1.4,
        log_extra_report="report log",
        log_extra_info="info log",
        custom_node_label="custom label",
        tag_ifTrue="test true tag",
        tag_ifFalse="test false tag",
    )
    # scales are set to make sure 3 components are true and 1 is false using the sample compnent table
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numTrue"] == 3
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numFalse"] == 1
    assert f"Node {selector.current_node_idx}" in selector.component_status_table

    # No components with "NotALabel" classification so nothing selected and no
    #   Node 1 column not created in component_status_table
    selector.current_node_idx = 1
    selector = selection_nodes.dec_left_op_right(
        selector,
        "accepted",
        "rejected",
        "NotAClassification",
        ">",
        "kappa",
        "rho",
    )
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numTrue"] == 0
    assert f"Node {selector.current_node_idx}" not in selector.component_status_table

    # Re-initializing selector so that it has components classificated as "provisional accept" again
    selector = sample_selector(options="provclass")
    # Test when left is a component_table_metric, & right is across_component_metric
    selector = selection_nodes.dec_left_op_right(
        selector,
        "accepted",
        "rejected",
        decide_comps,
        ">",
        "kappa",
        "test_elbow",
    )
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numTrue"] == 3
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numFalse"] == 1
    assert f"Node {selector.current_node_idx}" in selector.component_status_table

    # right component_table_metric, left cross_component_metric
    selector = sample_selector(options="provclass")
    selector = selection_nodes.dec_left_op_right(
        selector,
        "accepted",
        "rejected",
        decide_comps,
        ">",
        "test_elbow",
        "kappa",
    )
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numTrue"] == 1
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numFalse"] == 3
    assert f"Node {selector.current_node_idx}" in selector.component_status_table

    # left component_table_metric, right is a constant integer value
    selector = sample_selector(options="provclass")
    selector = selection_nodes.dec_left_op_right(
        selector,
        "accepted",
        "rejected",
        decide_comps,
        ">",
        "kappa",
        21,
    )
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numTrue"] == 3
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numFalse"] == 1
    assert f"Node {selector.current_node_idx}" in selector.component_status_table

    # right component_table_metric, left is a constant float value
    selector = sample_selector(options="provclass")
    selector = selection_nodes.dec_left_op_right(
        selector,
        "accepted",
        "rejected",
        decide_comps,
        ">",
        21.0,
        "kappa",
    )
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numTrue"] == 1
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numFalse"] == 3
    assert f"Node {selector.current_node_idx}" in selector.component_status_table

    # Testing combination of two statements. kappa>21 AND rho<13
    selector = sample_selector(options="provclass")
    selector = selection_nodes.dec_left_op_right(
        selector,
        "accepted",
        "rejected",
        decide_comps,
        "<",
        21.0,
        "kappa",
        left2="rho",
        op2="<",
        right2=14,
    )
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numTrue"] == 2
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numFalse"] == 2
    assert f"Node {selector.current_node_idx}" in selector.component_status_table


def test_dec_left_op_right_fails():
    """tests for calls to dec_left_op_right that raise errors"""

    selector = sample_selector(options="provclass")
    decide_comps = "provisional accept"

    # Raise error for left string that is not a metric
    selector = sample_selector(options="provclass")
    with pytest.raises(ValueError):
        selection_nodes.dec_left_op_right(
            selector,
            "accepted",
            "rejected",
            decide_comps,
            ">",
            "NotAMetric",
            21,
        )

    # Raise error for right string that is not a metric
    selector = sample_selector(options="provclass")
    with pytest.raises(ValueError):
        selection_nodes.dec_left_op_right(
            selector,
            "accepted",
            "rejected",
            decide_comps,
            ">",
            21,
            "NotAMetric",
        )

    # Raise error for invalid operator
    selector = sample_selector(options="provclass")
    with pytest.raises(ValueError):
        selection_nodes.dec_left_op_right(
            selector,
            "accepted",
            "rejected",
            decide_comps,
            "><",
            "kappa",
            21,
        )


def test_dec_variance_lessthan_thresholds_smoke():
    """Smoke tests for dec_variance_lessthan_thresholds"""

    selector = sample_selector(options="provclass")
    decide_comps = "provisional accept"

    # Outputs just the metrics used in this function {"variance explained"}
    used_metrics = selection_nodes.dec_variance_lessthan_thresholds(
        selector, "accepted", "rejected", decide_comps, only_used_metrics=True
    )
    assert len(used_metrics - {"variance explained"}) == 0

    # Standard execution where with all extra logging code and options changed from defaults
    selector = selection_nodes.dec_variance_lessthan_thresholds(
        selector,
        "accepted",
        "rejected",
        decide_comps,
        var_metric="normalized variance explained",
        single_comp_threshold=0.05,
        all_comp_threshold=0.09,
        log_extra_report="report log",
        log_extra_info="info log",
        custom_node_label="custom label",
        tag_ifTrue="test true tag",
        tag_ifFalse="test false tag",
    )
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numTrue"] == 1
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numFalse"] == 3
    assert f"Node {selector.current_node_idx}" in selector.component_status_table

    # No components with "NotALabel" classification so nothing selected and no
    #   Node 1 column not created in component_status_table
    selector.current_node_idx = 1
    selector = selection_nodes.dec_variance_lessthan_thresholds(
        selector, "accepted", "rejected", "NotAClassification"
    )
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numTrue"] == 0
    assert f"Node {selector.current_node_idx}" not in selector.component_status_table

    # Running without specifying logging text generates internal text
    selector = sample_selector(options="provclass")
    selector = selection_nodes.dec_variance_lessthan_thresholds(
        selector, "accepted", "rejected", decide_comps
    )
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numTrue"] == 0
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numFalse"] == 4
    assert f"Node {selector.current_node_idx}" in selector.component_status_table


def test_calc_kappa_rho_elbows_kundu():
    """Smoke tests for calc_kappa_rho_elbows_kundu"""

    # Standard use of this function requires some components to be "unclassified"
    selector = sample_selector(options="unclass")
    decide_comps = "all"

    # Outputs just the metrics used in this function {"variance explained"}
    used_metrics = selection_nodes.calc_kappa_rho_elbows_kundu(
        selector, decide_comps, only_used_metrics=True
    )
    assert len(used_metrics - {"kappa", "rho"}) == 0

    # Standard call to this function.
    selector = selection_nodes.calc_kappa_rho_elbows_kundu(
        selector,
        decide_comps,
        log_extra_report="report log",
        log_extra_info="info log",
        custom_node_label="custom label",
    )
    calc_cross_comp_metrics = {"kappa_elbow_kundu", "rho_elbow_kundu", "varex_upper_p"}
    output_calc_cross_comp_metrics = set(
        selector.tree["nodes"][selector.current_node_idx]["outputs"]["calc_cross_comp_metrics"]
    )
    # Confirming the indended metrics are added to outputs and they have non-zero values
    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["kappa_elbow_kundu"] > 0
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["rho_elbow_kundu"] > 0
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["varex_upper_p"] > 0

    # Run warning logging code for if any of the cross_component_metrics already existed and would be over-written
    selector = sample_selector(options="unclass")
    selector.cross_component_metrics["kappa_elbow_kundu"] = 1
    selector.cross_component_metrics["rho_elbow_kundu"] = 1
    selector.cross_component_metrics["varex_upper_p"] = 1
    decide_comps = "all"
    selector = selection_nodes.calc_kappa_rho_elbows_kundu(
        selector,
        decide_comps,
        log_extra_report="report log",
        log_extra_info="info log",
        custom_node_label="custom label",
    )
    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert selector.cross_component_metrics["kappa_elbow_kundu"] > 2
    assert selector.cross_component_metrics["rho_elbow_kundu"] > 2
    assert selector.cross_component_metrics["varex_upper_p"] > 2

    # Run with kappa_only==True
    selector = sample_selector(options="unclass")
    selector = selection_nodes.calc_kappa_rho_elbows_kundu(selector, decide_comps, kappa_only=True)
    calc_cross_comp_metrics = {"kappa_elbow_kundu", "varex_upper_p"}
    output_calc_cross_comp_metrics = set(
        selector.tree["nodes"][selector.current_node_idx]["outputs"]["calc_cross_comp_metrics"]
    )
    # Confirming the indended metrics are added to outputs and they have non-zero values
    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["kappa_elbow_kundu"] > 0
    assert "rho_elbow_kundu" not in selector.tree["nodes"][selector.current_node_idx]["outputs"]
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["varex_upper_p"] > 0

    # Run with rho_only==True
    selector = sample_selector(options="unclass")
    selector = selection_nodes.calc_kappa_rho_elbows_kundu(selector, decide_comps, rho_only=True)
    calc_cross_comp_metrics = {"rho_elbow_kundu", "varex_upper_p"}
    output_calc_cross_comp_metrics = set(
        selector.tree["nodes"][selector.current_node_idx]["outputs"]["calc_cross_comp_metrics"]
    )
    # Confirming the indended metrics are added to outputs and they have non-zero values
    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["rho_elbow_kundu"] > 0
    assert "kappa_elbow_kundu" not in selector.tree["nodes"][selector.current_node_idx]["outputs"]
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["varex_upper_p"] > 0

    # Should run normally with both kappa_only and rho_only==True
    selector = sample_selector(options="unclass")
    selector = selection_nodes.calc_kappa_rho_elbows_kundu(
        selector, decide_comps, kappa_only=True, rho_only=True
    )
    calc_cross_comp_metrics = {"kappa_elbow_kundu", "rho_elbow_kundu", "varex_upper_p"}
    output_calc_cross_comp_metrics = set(
        selector.tree["nodes"][selector.current_node_idx]["outputs"]["calc_cross_comp_metrics"]
    )
    # Confirming the indended metrics are added to outputs and they have non-zero values
    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["kappa_elbow_kundu"] > 0
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["rho_elbow_kundu"] > 0
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["varex_upper_p"] > 0

    # Log without running if no components of class decide_comps or no components
    #  classified as "unclassified" are in the component table
    selector = sample_selector()
    selector = selection_nodes.calc_kappa_rho_elbows_kundu(selector, "NotAClassification")
    calc_cross_comp_metrics = {"kappa_elbow_kundu", "rho_elbow_kundu", "varex_upper_p"}
    assert (
        selector.tree["nodes"][selector.current_node_idx]["outputs"]["kappa_elbow_kundu"] == None
    )
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["rho_elbow_kundu"] == None
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["varex_upper_p"] == None


def test_dec_classification_exists_smoke():
    """Smoke tests for dec_classification_exists"""

    selector = sample_selector(options="unclass")
    decide_comps = ["unclassified", "provisional accept"]

    # Outputs just the metrics used in this function {"variance explained"}
    used_metrics = selection_nodes.dec_classification_exists(
        selector,
        "rejected",
        decide_comps,
        class_comp_exists="provisional accept",
        only_used_metrics=True,
    )
    assert len(used_metrics) == 0

    # Standard execution where with all extra logging code and options changed from defaults
    selector = selection_nodes.dec_classification_exists(
        selector,
        "accepted",
        decide_comps,
        class_comp_exists="provisional accept",
        log_extra_report="report log",
        log_extra_info="info log",
        custom_node_label="custom label",
        tag_ifTrue="test true tag",
        tag_ifFalse="test false tag",
    )
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numTrue"] == 0
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numFalse"] == 0
    # During normal execution, it will find provionally accepted components
    #  and do nothing so another node isn't created
    assert f"Node {selector.current_node_idx}" not in selector.component_status_table

    # No components with "NotALabel" classification so nothing selected and no
    #   Node 1 column not created in component_status_table
    # Running without specifying logging text generates internal text
    selector.current_node_idx = 1
    selector = selection_nodes.dec_classification_exists(
        selector,
        "accepted",
        "NotAClassification",
        class_comp_exists="provisional accept",
    )
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numTrue"] == 0
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numTrue"] == 0
    assert f"Node {selector.current_node_idx}" not in selector.component_status_table

    # Other normal state is to change classifications when there are
    # no components with class_comp_exists. Since the component_table
    # initialized with sample_selector as not "provisional reject"
    # components, using that for class_comp_exists
    selector = sample_selector()
    decide_comps = "accepted"
    selector = selection_nodes.dec_classification_exists(
        selector,
        "changed accepted",
        decide_comps,
        class_comp_exists="provisional reject",
        tag_ifTrue="test true tag",
        tag_ifFalse="test false tag",
    )
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numTrue"] == 17
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numFalse"] == 0
    assert f"Node {selector.current_node_idx}" in selector.component_status_table


def test_calc_varex_upper_thresh_smoke():
    """Smoke tests for calc_varex_upper_thresh"""

    # Standard use of this function requires some components to be "provisional accept"
    selector = sample_selector(options="provclass")
    decide_comps = "provisional accept"

    # Outputs just the metrics used in this function {"variance explained"}
    used_metrics = selection_nodes.calc_varex_upper_thresh(
        selector, decide_comps, only_used_metrics=True
    )
    assert len(used_metrics - set(["variance explained"])) == 0

    # Standard call to this function.
    selector = selection_nodes.calc_varex_upper_thresh(
        selector,
        decide_comps,
        high_perc=90,
        log_extra_report="report log",
        log_extra_info="info log",
        custom_node_label="custom label",
    )
    calc_cross_comp_metrics = {"varex_upper_thresh", "high_perc"}
    output_calc_cross_comp_metrics = set(
        selector.tree["nodes"][selector.current_node_idx]["outputs"]["calc_cross_comp_metrics"]
    )
    # Confirming the indended metrics are added to outputs and they have non-zero values
    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["varex_upper_thresh"] > 0
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["high_perc"] == 90

    # Run warning logging code for if any of the cross_component_metrics already existed and would be over-written
    selector = sample_selector(options="provclass")
    selector.cross_component_metrics["varex_upper_thresh"] = 1
    selector.cross_component_metrics["high_perc"] = 1
    decide_comps = "provisional accept"
    selector = selection_nodes.calc_varex_upper_thresh(
        selector,
        decide_comps,
        log_extra_report="report log",
        log_extra_info="info log",
        custom_node_label="custom label",
    )
    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["varex_upper_thresh"] > 0
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["high_perc"] == 90

    # Run with high_perc already defined (and set to None here)
    selector = sample_selector(options="provclass")
    selector.cross_component_metrics["high_perc"] = 80
    selector = selection_nodes.calc_varex_upper_thresh(
        selector,
        decide_comps,
        high_perc=None,
    )
    calc_cross_comp_metrics = {"varex_upper_thresh"}
    output_calc_cross_comp_metrics = set(
        selector.tree["nodes"][selector.current_node_idx]["outputs"]["calc_cross_comp_metrics"]
    )
    # Confirming the indended metrics are added to outputs and they have non-zero values
    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["varex_upper_thresh"] > 0

    # Raise error if high_perc == None, but not already defined
    selector = sample_selector(options="provclass")
    with pytest.raises(ValueError):
        selector = selection_nodes.calc_varex_upper_thresh(
            selector,
            decide_comps,
            high_perc=None,
        )

    # Log without running if no components of decide_comps are in the component table
    selector = sample_selector()
    selector = selection_nodes.calc_varex_upper_thresh(selector, decide_comps="NotAClassification")
    assert (
        selector.tree["nodes"][selector.current_node_idx]["outputs"]["varex_upper_thresh"] == None
    )
    # high_perc doesn't depend on components and is assigned
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["high_perc"] == 90


def test_calc_extend_factor_smoke():
    """Smoke tests for calc_extend_factor"""

    selector = sample_selector()

    # Outputs just the metrics used in this function {""}
    used_metrics = selection_nodes.calc_extend_factor(selector, only_used_metrics=True)
    assert used_metrics == {""}

    # Standard call to this function.
    selector = selection_nodes.calc_extend_factor(
        selector,
        log_extra_report="report log",
        log_extra_info="info log",
        custom_node_label="custom label",
    )
    calc_cross_comp_metrics = {"extend_factor"}
    output_calc_cross_comp_metrics = set(
        selector.tree["nodes"][selector.current_node_idx]["outputs"]["calc_cross_comp_metrics"]
    )
    # Confirming the indended metrics are added to outputs and they have non-zero values
    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["extend_factor"] > 0

    # Run warning logging code for if any of the cross_component_metrics already existed and would be over-written
    selector = sample_selector()
    selector.cross_component_metrics["extend_factor"] = 1.0
    selector = selection_nodes.calc_extend_factor(selector)

    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["extend_factor"] > 0

    # Run with extend_factor defined as an input
    selector = sample_selector()
    selector = selection_nodes.calc_extend_factor(selector, extend_factor=1.2)

    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["extend_factor"] == 1.2


def test_max_good_meanmetricrank_smoke():
    """Smoke tests for calc_max_good_meanmetricrank"""

    # Standard use of this function requires some components to be "provisional accept"
    selector = sample_selector("provclass")
    # This function requires "extend_factor" to already be defined
    selector.cross_component_metrics["extend_factor"] = 2.0

    # Outputs just the metrics used in this function {""}
    used_metrics = selection_nodes.calc_max_good_meanmetricrank(
        selector, "provisional accept", only_used_metrics=True
    )
    assert used_metrics == set()

    # Standard call to this function.
    selector = selection_nodes.calc_max_good_meanmetricrank(
        selector,
        "provisional accept",
        log_extra_report="report log",
        log_extra_info="info log",
        custom_node_label="custom label",
    )
    calc_cross_comp_metrics = {"max_good_meanmetricrank"}
    output_calc_cross_comp_metrics = set(
        selector.tree["nodes"][selector.current_node_idx]["outputs"]["calc_cross_comp_metrics"]
    )
    # Confirming the indended metrics are added to outputs and they have non-zero values
    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert (
        selector.tree["nodes"][selector.current_node_idx]["outputs"]["max_good_meanmetricrank"] > 0
    )

    # Run warning logging code for if any of the cross_component_metrics already existed and would be over-written
    selector = sample_selector("provclass")
    selector.cross_component_metrics["max_good_meanmetricrank"] = 10
    selector.cross_component_metrics["extend_factor"] = 2.0

    selector = selection_nodes.calc_max_good_meanmetricrank(selector, "provisional accept")

    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert (
        selector.tree["nodes"][selector.current_node_idx]["outputs"]["max_good_meanmetricrank"] > 0
    )

    # Raise an error if "extend_factor" isn't pre-defined
    selector = sample_selector("provclass")
    with pytest.raises(ValueError):
        selector = selection_nodes.calc_max_good_meanmetricrank(selector, "provisional accept")

    # Log without running if no components of decide_comps are in the component table
    selector = sample_selector()
    selector.cross_component_metrics["extend_factor"] = 2.0

    selector = selection_nodes.calc_max_good_meanmetricrank(selector, "NotAClassification")
    assert (
        selector.tree["nodes"][selector.current_node_idx]["outputs"]["max_good_meanmetricrank"]
        == None
    )