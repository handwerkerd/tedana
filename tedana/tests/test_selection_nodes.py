"""Tests for the tedana.selection.selection_nodes module."""
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

    # Standard execution where components are changed from "provisional accept" to "accepted"
    # And all extra logging code is run
    decide_comps = "provisional accept"
    new_classification = "accepted"
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
    assert f"Node {selector.current_node_idx}" in selector.component_status_table

    # Outputs just the metrics used in this function (nothing in this case)
    used_metrics = selection_nodes.manual_classify(
        selector, decide_comps, new_classification, only_used_metrics=True
    )
    assert used_metrics == set()

    # No components with a "NotALabel" classificaiton so nothing selected and no
    #   "Node 1 column created"
    selector.current_node_idx = 1
    selector = selection_nodes.manual_classify(selector, "NotALabel", new_classification)
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


def test_dec_left_op_right_smoke():
    """Smoke tests for dec_left_op_right"""

    selector = sample_selector(options="provclass")

    # Standard execution where components are changed from "provisional accept" to "accepted"
    # And all extra logging code is run
    decide_comps = "provisional accept"

    used_metrics = selection_nodes.dec_left_op_right(
        selector, "accepted", "rejected", decide_comps, ">", "kappa", "rho", only_used_metrics=True
    )
    # Outputs just the metrics used in this function {"kappa", "rho"}
    assert len(used_metrics - {"kappa", "rho"}) == 0

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
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numTrue"] == 3
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numFalse"] == 1
    assert f"Node {selector.current_node_idx}" in selector.component_status_table

    # left component_table_metric, right cross_component_metric
    selector = sample_selector(options="provclass")
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

    # left component_table_metric, right constant
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

    # right component_table_metric, left constant
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

    # Raise error for invalie operator
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
