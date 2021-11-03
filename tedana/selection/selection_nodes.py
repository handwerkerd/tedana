"""
Functions that will be used as steps in a decision tree
"""
import logging
import numpy as np
import pandas as pd

# from scipy import stats

from scipy.stats import scoreatpercentile
from tedana.stats import getfbounds
from tedana.selection._utils import (
    confirm_metrics_exist,
    selectcomps2use,
    log_decision_tree_step,
    change_comptable_classifications,
    getelbow,
    create_dnode_outputs,
    get_extend_factor,
    kappa_elbow_kundu,
    get_new_meanmetricrank,
    prev_classified_comps,
)


# clean_dataframe, new_decision_node_info,
LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")
RefLGR = logging.getLogger("REFERENCES")

decision_docs = {
    "comptable": """\
comptable : (C x M) :obj:`pandas.DataFrame`
    Component metric table. One row for each component, with a column for
    each metric. The index should be the component number.\
""",
    "decision_node_idx": """\
decison_node_idx : :obj: `int`
    The decision tree function are run as part of an ordered list.
    This is the positional index for when this function has been run
    as part of this list.\
""",
    "ifTrue": """\
ifTrue : :obj:`str`
    If the condition in this step is true, give the component
    the label in this string. Options are 'accepted', 'rejected',
    'nochange', or intermediate_classification labels predefined in the
    decision tree
    If 'nochange' then don't change the current component classification\
""",
    "ifFalse": """\
ifFalse: :obj:`str`
    If the condition in this step is false, give the component the label
    in this string. Same options as ifTrue\
""",
    "decide_comps": """\
decide_comps: :obj:`str` or :obj:`list[str]` or :obj:`int` or :obj:`list[int]`
    If this is string or a list of strings describing what classifications
    of components to operate on, using the same labels as in ifTrue.
    For example: decide_comps='unclassified' means to operate only on
    unclassified components. The label 'all' will operate on all
    components regardess of classification.
    If this is an int or a list of int, operate on components with the
    listed integer indices. For example: [0 1 5] will operate on 3 components
    regardless of what their current classifications are.\
""",
    "log_extra": """\
log_extra_report, log_extra_info: :obj:`str`
    Text for each function call is automatically placed in the logger output
    In addition to that text, the text in these these strings will also be
    included in the logger with the report or info codes respectively.
    These might be useful to give a narrative explanation of why a step was
    parameterized a certain way. default="" (no extra logging)\
""",
    "only_used_metrics": """\
only_used_metrics: :obj:`bool`
    If true, this function will only return the names of the comptable metrics
    that will be used when this function is fully run. default=False\
""",
    "custom_node_label": """\
custom_node_label: :obj:`str`
    A brief label for what happens in this node that can be used in a decision
tree summary table or flow chart. If custom_node_label is not empty, then the
text in this parameter is used instead of the text would be automatically
assigned within the function call default=""\
""",
    "basicreturns": """\
comptable: (C x M) :obj:`pandas.DataFrame`
    Component metric table. One row for each component, with a column for
    each metric. The index should be the component number.
    Labels in the 'classifications' for components initially labeled in
    decide_comps may change depending on the the ifTrue and ifFalse instructions.
    When a classification changes, the 'rationale' column is appended to include
    and additional decision node index and change. For example, if this function
    is the 5th decision node run and a component is reclassified as 'accepted',
    the string in 'rationale' is appended to include '5: accepted;'
    comptable is only only returned if only_used_metrics=False
dnode_outputs: :obj:`dict`
    Several parameters that should be output from each decision node function
    in a dictionary under the key 'outputs' When a function is run as output
    as part of the decision tree class, this output will be added to the dictionary
    with parameters that called the function with all the outputs under the 'outputs'
    key. dnode_outputs includes the following fields\
    used_metrics: :obj: `list[str]`
        A list of all metrics from the comptable header used within this function.
        Note, this must be a list even if only one metric is used
    node_label: :obj: `str`
        A brief label for what happens in this node that can be used in a decision
    tree summary table or flow chart. This is defined in the function unless
    custom_node_label is not empty. In that case, node_label=custom_node_label
    numTrue, numFalse: :obj: `int`
        The number of components that were classified as true or false respectively
    in this decision tree step.\
""",
    "n_echos": """\
n_echos: :obj:`int`
    The number of echos in the multi-echo data
        \
""",
    "n_vols": """\
n_vols: :obj:`int`
    The number of volumes (time points) in the fMRI data
        \
""",
    "extend_factor": """\
extend_factor: :obj:`float`
    A scaler used to set the threshold for the mean rank metric
        \
        """,
    "restrict_factor": """\
restrict_factor: :obj:`float`
    A scaler used to set the threshold for the mean rank metric
        \
        """,
    "prev_X_steps": """\
prev_X_steps: :obj:`int`
    Search for components with a classification label in the current or the previous X steps in
    the decision tree
        \
        """,
    "kappa_elbow": """\
kappa_elbow: :obj:`float`
    The kappa threshold below which components are less likely to contain T2* signal
        \
""",
    "rho_elbow": """\
rho_elbow: :obj:`float`
    The rho threshold above which components are likely to contain nontrivial S0 signal
        \
""",
}


def manual_classify(
    selector,
    decide_comps,
    new_classification,
    clear_rationale=False,
    log_extra_report="",
    log_extra_info="",
    custom_node_label="",
    only_used_metrics=False,
):
    """
    Explicitly assign a classifictation, defined in iffrue,
    to all the components in decide_comps. This was designed
    with three use cases in mind:
    1. Set the classifictions of all components to unclassified
    for the first node of a decision tree.
    clear_rationale=True is recommended for this use case
    2. Shift all components between classifications, such as
    provisionalaccept to accepted for the penultimate node in the
    decision tree.
    3. Manually re-classify components by number based on user
    observations.

    Parameters
    ----------
    {comptable}
    {decision_node_idx}
    {decide_comps}
    new_classification: :obj: `str`
        Assign all components identified in decide_comps the classification
        in new_classification. Options are 'unclassified', 'accepted',
        'rejected', or intermediate_classification labels predefined in the
        decision tree
    clear_rationale: :obj: `bool`
        If True, reset all values in the 'rationale' column to empty strings
        If False, do nothing
    {log_extra}
    {custom_node_label}
    {only_used_metrics}

    Returns
    -------
    {basicreturns}

    Note
    ----
    Unlike other decision node functions, ifTrue and ifFalse are not inputs
    since the same classification is assigned to all components listed in
    decide_comps
    """

    # predefine all outputs that should be logged
    outputs = {
        "decision_node_idx": selector.current_node_idx,
        "used_metrics": set(),
        "node_label": None,
        "numTrue": None,
        "numFalse": None,
    }

    if only_used_metrics:
        return outputs["used_metrics"]

    ifTrue = new_classification
    ifFalse = "nochange"

    function_name_idx = "Step {}: manual_classify".format((selector.current_node_idx))
    if custom_node_label:
        outputs["node_label"] = custom_node_label
    else:
        outputs["node_label"] = "Set " + str(decide_comps) + " to " + new_classification

    if log_extra_info:
        LGR.info(log_extra_info)
    if log_extra_report:
        RepLGR.info(log_extra_report)

    comps2use, component_table = selectcomps2use(selector, decide_comps)

    if comps2use is None:
        log_decision_tree_step(function_name_idx, comps2use, decide_comps=decide_comps)
        outputs["numTrue"] = 0
        outputs["numFalse"] = 0
    else:
        decision_boolean = pd.Series(True, index=comps2use)
        selector = change_comptable_classifications(
            selector, ifTrue, ifFalse, decision_boolean
        )
        outputs["numTrue"] = decision_boolean.sum()
        outputs["numFalse"] = np.logical_not(decision_boolean).sum()
        # print(('numTrue={}, numFalse={}, numcomps2use={}'.format(
        #    numTrue, numFalse, len(comps2use))))
        log_decision_tree_step(
            function_name_idx,
            comps2use,
            numTrue=outputs["numTrue"],
            numFalse=outputs["numFalse"],
            ifTrue=ifTrue,
            ifFalse=ifFalse,
        )

    if clear_rationale:
        component_table["rationale"] = ""
        LGR.info(
            function_name_idx
            + " component classification 'rationale' values are set to empty strings"
        )

    selector.nodes[selector.current_node_idx]["outputs"] = outputs

    return selector


manual_classify.__doc__ = manual_classify.__doc__.format(**decision_docs)


def left_op_right(
    selector,
    ifTrue,
    ifFalse,
    decide_comps,
    op,
    left,
    right,
    right_scale=1,
    left_scale=1,
    log_extra_report="",
    log_extra_info="",
    custom_node_label="",
    only_used_metrics=False,
):
    """
    Tests a relationship between (left_scale*)left and (right_scale*right)
    using an operator, like >, defined with op
    This can be used to directly compare any 2 metrics and use that info
    to change component classification. If either metric is a number,
    this can also compare a metric against a fixed threshold.

    Parameters
    ----------
    {comptable}
    {decision_node_idx}
    {ifTrue}
    {ifFalse}
    {decide_comps}

    op: :ojb:`str`
        Must be one of: ">", ">=", "==", "<=", "<"
        Applied the user defined operator to left op right
    left, right: :obj:`str` or :obj:`float`
        The labels for the two metrics to be used for comparision.
        for example: left='kappa', right='rho' and op='>' means this
        function will test kappa>rho. One of the two can also be a number.
        In that case a metric would be compared against a fixed threshold.
        For example left='T2fitdiff_invsout_ICAmap_Tstat', right=0, and op='>'
        means this function will test T2fitdiff_invsout_ICAmap_Tstat>0
    left_scale, right_scale: :obj:`float`, optional
            Multiply the left or right metrics value by a constant. For example
            if left='kappa', right='rho', right_scale=2, and op='>' this tests
            kappa>(2*rho). default=1
    {log_extra}
    {custom_node_label}
    {only_used_metrics}

    Returns
    -------
    {basicreturns}
    """

    # predefine all outputs that should be logged
    outputs = {
        "decision_node_idx": selector.current_node_idx,
        "used_metrics": set(),
        "node_label": None,
        "numTrue": None,
        "numFalse": None,
    }

    if isinstance(left, str):
        outputs["used_metrics"].update([left])
    if isinstance(right, str):
        outputs["used_metrics"].update([right])
    if only_used_metrics:
        return outputs["used_metrics"]

    legal_ops = (">", ">=", "==", "<=", "<")
    if op not in legal_ops:
        raise ValueError(f"{op} is not a binary comparison operator, like > or <")

    function_name_idx = f"Step {selector.current_node_idx}: left_op_right"
    if custom_node_label:
        outputs["node_label"] = custom_node_label
    else:
        if left_scale == 1:
            tmp_left_scale = ""
        else:
            tmp_left_scale = f"{left_scale}*"
        if right_scale == 1:
            tmp_right_scale = ""
        else:
            tmp_right_scale = f"{right_scale}*"
        outputs["node_label"] = f"{tmp_left_scale}{left}{op}{tmp_right_scale}{right}"

    # Might want to add additional default logging to functions here
    # The function input will be logged before the function call
    if log_extra_info:
        LGR.info(log_extra_info)
    if log_extra_report:
        RepLGR.info(log_extra_report)

    comps2use, component_table = selectcomps2use(selector, decide_comps)

    confirm_metrics_exist(
        component_table, outputs["used_metrics"], function_name=function_name_idx
    )

    if comps2use is None:
        log_decision_tree_step(function_name_idx, comps2use, decide_comps=decide_comps)
        outputs["numTrue"] = 0
        outputs["numFalse"] = 0
    else:
        if isinstance(left, str):
            val1 = component_table.loc[comps2use, left]
        else:
            val1 = left  # should be a fixed number
        if isinstance(right, str):
            val2 = component_table.loc[comps2use, right]
        else:
            val2 = right  # should be a fixed number
        decision_boolean = eval(f"(left_scale*val1) {op} (right_scale * val2)")

        selector = change_comptable_classifications(
            selector, ifTrue, ifFalse, decision_boolean
        )
        outputs["numTrue"] = np.asarray(decision_boolean).sum()
        outputs["numFalse"] = np.logical_not(decision_boolean).sum()
        # print(('numTrue={}, numFalse={}, numcomps2use={}'.format(
        #    numTrue, numFalse, len(comps2use))))
        log_decision_tree_step(
            function_name_idx,
            comps2use,
            numTrue=outputs["numTrue"],
            numFalse=outputs["numFalse"],
            ifTrue=ifTrue,
            ifFalse=ifFalse,
        )

    selector.nodes[selector.current_node_idx]["outputs"] = outputs

    return selector


left_op_right.__doc__ = left_op_right.__doc__.format(**decision_docs)


def variance_lessthan_thresholds(
    selector,
    ifTrue,
    ifFalse,
    decide_comps,
    var_metric="varexp",
    single_comp_threshold=0.1,
    all_comp_threshold=1.0,
    log_extra_report="",
    log_extra_info="",
    custom_node_label="",
    only_used_metrics=False,
):
    """
    Finds components with variance<single_comp_threshold.
    If the sum of the variance for all components that meet this criteria
    is greater than all_comp_threshold then keep the lowest variance
    components so that the sum of their variances is less than all_comp_threshold

    Parameters
    ----------
    {comptable}
    {decision_node_idx}
    {ifTrue}
    {ifFalse}
    {decide_comps}
    var_metric: :obj:`str`
        The name of the metric in comptable for variance. default=varexp
        This is an option so that it is possible to set this to normvarexp
        or some other variance measure
    single_comp_threshold: :obj:`float`
        The threshold for which all components need to have lower variance
    all_comp_threshold: :obj: `float`
        The threshold for which the sum of all components<single_comp_threshold
        needs to be under
    {log_extra}
    {custom_node_label}
    {only_used_metrics}

    Returns
    -------
    {basicreturns}
    """

    outputs = {
        "decision_node_idx": selector.current_node_idx,
        "used_metrics": set([var_metric]),
        "node_label": None,
        "numTrue": None,
        "numFalse": None,
    }

    if only_used_metrics:
        return outputs["used_metrics"]

    function_name_idx = "Step {}: variance_lt_thresholds".format(
        selector.current_node_idx
    )
    if custom_node_label:
        outputs["node_label"] = custom_node_label
    else:
        outputs["node_label"] = ("{}<{}. All variance<{}").format(
            outputs["used_metrics"], single_comp_threshold, all_comp_threshold
        )

    if log_extra_info:
        LGR.info(log_extra_info)
    if log_extra_report:
        RepLGR.info(log_extra_report)

    comps2use, component_table = selectcomps2use(selector, decide_comps)
    metrics_exist, missing_metrics = confirm_metrics_exist(
        component_table, outputs["used_metrics"], function_name=function_name_idx
    )

    if comps2use is None:
        log_decision_tree_step(function_name_idx, comps2use, decide_comps=decide_comps)
        outputs["numTrue"] = 0
        outputs["numFalse"] = 0
    else:
        variance = component_table.loc[comps2use, var_metric]
        decision_boolean = variance < single_comp_threshold
        # if all the low variance components sum above all_comp_threshold
        # keep removing the highest remaining variance component until
        # the sum is below all_comp_threshold. This is an inefficient
        # way to do this, but it works & should never cause an infinite loop
        if variance[decision_boolean].sum() > all_comp_threshold:
            while variance[decision_boolean].sum() > all_comp_threshold:
                cutcomp = variance[decision_boolean].idxmax
                decision_boolean[cutcomp] = False
        selector = change_comptable_classifications(
            selector, ifTrue, ifFalse, decision_boolean
        )
        outputs["numTrue"] = np.asarray(decision_boolean).sum()
        outputs["numFalse"] = np.logical_not(decision_boolean).sum()
        # print(('numTrue={}, numFalse={}, numcomps2use={}'.format(
        #    numTrue, numFalse, len(comps2use))))
        log_decision_tree_step(
            function_name_idx,
            comps2use,
            numTrue=outputs["numTrue"],
            numFalse=outputs["numFalse"],
            ifTrue=ifTrue,
            ifFalse=ifFalse,
        )

    selector.nodes[selector.current_node_idx]["outputs"] = outputs
    return selector


variance_lessthan_thresholds.__doc__ = variance_lessthan_thresholds.__doc__.format(
    **decision_docs
)


def kappa_rho_elbow_cutoffs_kundu(
    selector,
    ifTrue,
    ifFalse,
    decide_comps,
    n_echos,
    log_extra_report="",
    log_extra_info="",
    custom_node_label="",
    only_used_metrics=False,
    kappa_only=False,
    rho_only=False,
):
    """
    Calculates 'elbows' for kappa and rho values across compnents and thresholds
    on kappa>kappa_elbow & rho<rho_elbow

    Parameters
    ----------
    {comptable}
    {decision_node_idx}
    {ifTrue}
    {ifFalse}
    {decide_comps}
    {n_echos}
    {log_extra}
    {custom_node_label}
    {only_used_metrics}
    kappa_only: :obj:`bool`, optional
            Only use the kappa>kappa_elbow threshold. default=False
    rho_only: :obj:`bool`, optional
            Only use the rho>rho_elbow threshold. default=False


    Returns
    -------
    {basicreturns}
    dnode_ouputs also contains:
    {n_echos}
    {kappa_elbow}
    {rho_elbow}

    Note
    ----
    This script is currently hard coded for a specific way to calculate kappa and rho elbows
    based on the method by Kundu in the MEICA v2.7 code. Another elbow calculation would
    require a distinct function. Ideally, there can be one elbow function can allows for
    some more flexible options
    """

    # If kappa_only or rho_only is true kappa or rho might not actually be
    # used, but, as of now, both are required to run this function

    outputs = {
        "decision_node_idx": selector.current_node_idx,
        "used_metrics": set(["kappa", "rho"]),
        "node_label": None,
        "numTrue": None,
        "numFalse": None,
        "n_echos": n_echos,
        "varex_upper_p": None,
        "kappa_elbow": None,
        "rho_elbow": None,
        "kappa_only": kappa_only,
        "rho_only": rho_only,
    }

    if only_used_metrics:
        return outputs["used_metrics"]

    function_name_idx = "Step {}: kappa_rho_elbow_cutoffs_kundu".format(
        selector.current_node_idx
    )
    if custom_node_label:
        outputs["node_label"] = custom_node_label
    else:
        if kappa_only:
            outputs["node_label"] = "Kappa Elbow Thresholding"
        elif rho_only:
            outputs["node_label"] = "Rho Elbow Thresholding"
        else:
            outputs["node_label"] = "Kappa&Rho Elbow Thresholding"

    LGR.info(
        "Note: This matches the elbow selecton criteria in Kundu's MEICA v2.7"
        " except there is a variance threshold that is used for the rho criteria that "
        "really didn't make sense and is being excluded."
    )

    if log_extra_info:
        LGR.info(log_extra_info)
    if log_extra_report:
        RepLGR.info(log_extra_report)

    comps2use, component_table = selectcomps2use(selector, decide_comps)
    metrics_exist, missing_metrics = confirm_metrics_exist(
        component_table, outputs["used_metrics"], function_name=function_name_idx
    )

    unclassified_comps2use = selectcomps2use(selector, "unclassified")[0]

    if (comps2use is None) or (unclassified_comps2use is None):
        if comps2use is None:
            log_decision_tree_step(
                function_name_idx, comps2use, decide_comps=decide_comps
            )
        if unclassified_comps2use is None:
            log_decision_tree_step(
                function_name_idx, comps2use, decide_comps="unclassified"
            )
        outputs["numTrue"] = 0
        outputs["numFalse"] = 0
    else:
        outputs["kappa_elbow"] = kappa_elbow_kundu(component_table, n_echos)

        # The first elbow used to be for rho values of the unclassified components
        # excluding a few based on differences of variance. Now it's all unclassified
        # components
        # Upper limit for variance explained is median across components with high
        # Kappa values. High Kappa is defined as Kappa above Kappa elbow.
        f05, _, f01 = getfbounds(n_echos)
        outputs["varex_upper_p"] = np.median(
            component_table.loc[
                component_table["kappa"]
                > getelbow(component_table["kappa"], return_val=True),
                "variance explained",
            ]
        )

        ncls = unclassified_comps2use.copy()
        for i_loop in range(3):
            temp_comptable = component_table.loc[ncls].sort_values(
                by=["variance explained"], ascending=False
            )
            diff_vals = temp_comptable["variance explained"].diff(-1)
            diff_vals = diff_vals.fillna(0)
            ncls = temp_comptable.loc[diff_vals < outputs["varex_upper_p"]].index.values
        # kappa_elbow was altready calculated in kappa_elbow_kundu above
        # kappas_nonsig = comptable.loc[comptable["kappa"] < f01, "kappa"]
        # kappa_elbow = np.min(
        #     (
        #         getelbow(kappas_nonsig, return_val=True),
        #         getelbow(comptable["kappa"], return_val=True),
        #     )
        # )
        outputs["rho_elbow"] = np.mean(
            (
                getelbow(component_table.loc[ncls, "rho"], return_val=True),
                getelbow(component_table["rho"], return_val=True),
                f05,
            )
        )

        if kappa_only:
            decision_boolean = (
                component_table.loc[comps2use, "kappa"] >= outputs["kappa_elbow"]
            )
        elif rho_only:
            decision_boolean = (
                component_table.loc[comps2use, "rho"] < outputs["rho_elbow"]
            )
        else:
            decision_boolean = (
                component_table.loc[comps2use, "kappa"] >= outputs["kappa_elbow"]
            ) & (component_table.loc[comps2use, "rho"] < outputs["rho_elbow"])

        selector = change_comptable_classifications(
            selector, ifTrue, ifFalse, decision_boolean
        )
        outputs["numTrue"] = np.asarray(decision_boolean).sum()
        outputs["numFalse"] = np.logical_not(decision_boolean).sum()
        # print(('numTrue={}, numFalse={}, numcomps2use={}'.format(
        #        numTrue, numFalse, len(comps2use))))
        log_decision_tree_step(
            function_name_idx,
            comps2use,
            numTrue=outputs["numTrue"],
            numFalse=outputs["numFalse"],
            ifTrue=ifTrue,
            ifFalse=ifFalse,
        )

    selector.nodes[selector.current_node_idx]["outputs"] = outputs

    return selector


kappa_rho_elbow_cutoffs_kundu.__doc__ = kappa_rho_elbow_cutoffs_kundu.__doc__.format(
    **decision_docs
)


def classification_exists(
    comptable,
    decision_node_idx,
    ifTrue,
    ifFalse,
    decide_comps,
    class_comp_exists,
    log_extra_report="",
    log_extra_info="",
    custom_node_label="",
    only_used_metrics=False,
):
    """
    If there are not compontents with a classification specified in class_comp_exists,
    change the classification of all components in decide_comps
    Parameters
    ----------
    {comptable}
    {decision_node_idx}
    {ifTrue}
    {ifFalse}
    {decide_comps}
    class_comp_exists: :obj:`str` or :obj:`list[str]` or :obj:`int` or :obj:`list[int]`
        This has the same structure options as decide_comps. This function tests
        whether any components have the classifications defined in this variable.
    {log_extra}
    {custom_node_label}
    {only_used_metrics}

    Returns
    -------
    {basicreturns}

    """

    used_metrics = []
    if only_used_metrics:
        return used_metrics

    function_name_idx = "Step {}: classification_exists".format(decision_node_idx)
    if custom_node_label:
        node_label = custom_node_label
    else:
        node_label = "Change {} if {} doesn't exist".format(
            decide_comps, classification_exists
        )

    # Might want to add additional default logging to functions here
    # The function input will be logged before the function call
    if log_extra_info:
        LGR.info(log_extra_info)
    if log_extra_report:
        RepLGR.info(log_extra_report)

    comps2use = selectcomps2use(comptable, decide_comps)
    do_comps_exist = selectcomps2use(comptable, class_comp_exists)

    if comps2use is None:
        log_decision_tree_step(function_name_idx, comps2use, decide_comps=decide_comps)
        numTrue = 0
        numFalse = 0
    elif do_comps_exist is None:
        # should be false for all components
        decision_boolean = comptable.loc[comps2use, "component"] < -100
        comptable = change_comptable_classifications(
            comptable, ifTrue, ifFalse, decision_boolean, str(decision_node_idx)
        )
        numTrue = np.asarray(decision_boolean).sum()
        # numtrue should always be 0 in this situation
        numFalse = np.logical_not(decision_boolean).sum()
        # print(('numTrue={}, numFalse={}, numcomps2use={}'.format(
        #    numTrue, numFalse, len(comps2use))))
        log_decision_tree_step(
            function_name_idx,
            comps2use,
            numTrue=numTrue,
            numFalse=numFalse,
            ifTrue=ifTrue,
            ifFalse=ifFalse,
        )
    else:
        numTrue = len(comps2use)
        numFalse = 0
        log_decision_tree_step(
            function_name_idx,
            comps2use,
            numTrue=numTrue,
            numFalse=numFalse,
            ifTrue=ifTrue,
            ifFalse=ifFalse,
        )

    dnode_outputs = create_dnode_outputs(
        decision_node_idx, used_metrics, node_label, numTrue, numFalse
    )

    return comptable, dnode_outputs


def meanmetricrank_and_variance_greaterthan_thresh(
    comptable,
    decision_node_idx,
    ifTrue,
    ifFalse,
    decide_comps,
    n_vols,
    high_perc=90,
    extend_factor=None,
    log_extra_report="",
    log_extra_info="",
    custom_node_label="",
    only_used_metrics=False,
):
    """
    The 'mean metric rank' (formerly d_table) is the mean of rankings of 5 metrics:
        'kappa', 'dice_FT2', 'signal-noise_t',
        and 'countnoise', 'countsigFT2'
    For these 5 metrics, a lower rank (smaller number) is less likely to be
    T2* weighted.
    This function tests of meanmetricrank is above a threshold based on the number
    of provisionally accepted components & variance based on a threshold related
    to the variance of provisionally accepted components. This is indented to
    reject components that are greater than both of these thresholds

    Parameters
    ----------
    {comptable}
    {decision_node_idx}
    {ifTrue}
    {ifFalse}
    {decide_comps}
    {n_vols}
    high_perc: :obj:`int`
        A percentile threshold to apply to components to set the variance
        threshold. default=90
    {extend_factor}
    {log_extra}
    {custom_node_label}
    {only_used_metrics}

    Returns
    -------
    {basicreturns}
    dnode_ouputs also contains:
    num_prov_accept: :obj:`int`
        Number of provisionally accepted components
    max_good_meanmetricrank: :obj:`float`
        The threshold used meanmetricrank
    varex_threshold: :obj:`float`
        The threshold used for variance
    """

    used_metrics = ["d_table_score", "variance explained"]
    if only_used_metrics:
        return used_metrics

    function_name_idx = (
        "Step {}: meanmetricrank_and_variance_greaterthan_thresh".format(
            decision_node_idx
        )
    )
    if custom_node_label:
        node_label = custom_node_label
    else:
        node_label = "MeanRank & Variance Thresholding"

    if log_extra_info:
        LGR.info(log_extra_info)
    if log_extra_report:
        RepLGR.info(log_extra_report)

    metrics_exist, missing_metrics = confirm_metrics_exist(
        comptable, used_metrics, function_name=function_name_idx
    )

    comps2use = selectcomps2use(comptable, decide_comps)
    provaccept_comps2use = selectcomps2use(comptable, ["provisionalaccept"])
    if (comps2use is None) or (provaccept_comps2use is None):
        if comps2use is None:
            log_decision_tree_step(
                function_name_idx, comps2use, decide_comps=decide_comps
            )
        if provaccept_comps2use is None:
            log_decision_tree_step(
                function_name_idx, comps2use, decide_comps="provisionalaccept"
            )
        dnode_outputs = create_dnode_outputs(
            decision_node_idx, used_metrics, node_label, 0, 0
        )
    else:
        num_prov_accept = len(provaccept_comps2use)
        varex_upper_thresh = scoreatpercentile(
            comptable.loc[provaccept_comps2use, "variance explained"], high_perc
        )

        extend_factor = get_extend_factor(n_vols=n_vols, extend_factor=extend_factor)
        max_good_meanmetricrank = extend_factor * num_prov_accept

        decision_boolean1 = (
            comptable.loc[comps2use, "d_table_score"] > max_good_meanmetricrank
        )
        decision_boolean2 = (
            comptable.loc[comps2use, "variance explained"] > varex_upper_thresh
        )
        decision_boolean = decision_boolean1 & decision_boolean2

        comptable = change_comptable_classifications(
            comptable, ifTrue, ifFalse, decision_boolean, str(decision_node_idx)
        )
        numTrue = np.asarray(decision_boolean).sum()
        numFalse = np.logical_not(decision_boolean).sum()
        # print(('numTrue={}, numFalse={}, numcomps2use={}'.format(
        #    numTrue, numFalse, len(comps2use))))
        log_decision_tree_step(
            function_name_idx,
            comps2use,
            numTrue=numTrue,
            numFalse=numFalse,
            ifTrue=ifTrue,
            ifFalse=ifFalse,
        )

        dnode_outputs = create_dnode_outputs(
            decision_node_idx,
            used_metrics,
            node_label,
            numTrue,
            numFalse,
            num_prov_accept=num_prov_accept,
            varex_threshold=varex_upper_thresh,
            max_good_meanmetricrank=max_good_meanmetricrank,
            extend_factor=extend_factor,
        )

    return comptable, dnode_outputs


meanmetricrank_and_variance_greaterthan_thresh.__doc__ = (
    meanmetricrank_and_variance_greaterthan_thresh.__doc__.format(**decision_docs)
)


def lowvariance_highmeanmetricrank_lowkappa(
    comptable,
    decision_node_idx,
    ifTrue,
    ifFalse,
    decide_comps,
    n_echos,
    n_vols,
    low_perc=25,
    extend_factor=None,
    log_extra_report="",
    log_extra_info="",
    custom_node_label="",
    only_used_metrics=False,
):
    """
    Finds components with variance below a threshold,
    a mean metric rank above a threshold, and kappa below a threshold.
    This would typically be used to identify remaining components that would
    otherwise be rejected & put them in accept with a 'low variance' tag

    Parameters
    ----------
    {comptable}
    {decision_node_idx}
    {ifTrue}
    {ifFalse}
    {decide_comps}
    {n_echos}
    {n_vols}
    {extend_factor}
    {log_extra}
    {custom_node_label}
    {only_used_metrics}

    Returns
    -------
    {basicreturns}
    """

    used_metrics = ["variance explained", "kappa", "d_table_score"]
    if only_used_metrics:
        return used_metrics

    function_name_idx = "Step {}: lowvariance_highmeanmetricrank_lowkappa".format(
        decision_node_idx
    )
    if custom_node_label:
        node_label = custom_node_label
    else:
        node_label = "lowvar highmeanmetricrank lowkappa"

    if log_extra_info:
        LGR.info(log_extra_info)
    if log_extra_report:
        RepLGR.info(log_extra_report)
    metrics_exist, missing_metrics = confirm_metrics_exist(
        comptable, used_metrics, function_name=function_name_idx
    )

    comps2use = selectcomps2use(comptable, decide_comps)
    provaccept_comps2use = selectcomps2use(comptable, ["provisionalaccept"])

    if (comps2use is None) or (provaccept_comps2use is None):
        if comps2use is None:
            log_decision_tree_step(
                function_name_idx, comps2use, decide_comps=decide_comps
            )
        if provaccept_comps2use is None:
            log_decision_tree_step(
                function_name_idx, comps2use, decide_comps="provisionalaccept"
            )
        dnode_outputs = create_dnode_outputs(
            decision_node_idx, used_metrics, node_label, 0, 0
        )
    else:
        # low variance threshold
        varex_lower_thresh = scoreatpercentile(
            comptable.loc[provaccept_comps2use, "variance explained"], low_perc
        )
        db_low_varex = (
            comptable.loc[comps2use, "variance explained"] < varex_lower_thresh
        )

        # mean metric rank threshold
        num_prov_accept = len(provaccept_comps2use)
        extend_factor = get_extend_factor(n_vols=n_vols, extend_factor=extend_factor)
        max_good_meanmetricrank = extend_factor * num_prov_accept
        db_meanmetricrank = (
            comptable.loc[comps2use, "d_table_score"] < max_good_meanmetricrank
        )

        # kappa threshold
        kappa_elbow = kappa_elbow_kundu(comptable, n_echos)
        db_kappa = comptable.loc[comps2use, "kappa"] > kappa_elbow

        # combine the 3 thresholds
        decision_boolean = db_low_varex & db_meanmetricrank & db_kappa

        comptable = change_comptable_classifications(
            comptable, ifTrue, ifFalse, decision_boolean, str(decision_node_idx)
        )
        numTrue = np.asarray(decision_boolean).sum()
        numFalse = np.logical_not(decision_boolean).sum()
        # print(('numTrue={}, numFalse={}, numcomps2use={}'.format(
        #    numTrue, numFalse, len(comps2use))))

        log_decision_tree_step(
            function_name_idx,
            comps2use,
            numTrue=numTrue,
            numFalse=numFalse,
            ifTrue=ifTrue,
            ifFalse=ifFalse,
        )

        dnode_outputs = create_dnode_outputs(
            decision_node_idx,
            used_metrics,
            node_label,
            numTrue,
            numFalse,
            n_echos=n_echos,
            n_vols=n_vols,
            kappa_elbow=kappa_elbow,
            varex_threshold=varex_lower_thresh,
            max_good_meanmetricrank=max_good_meanmetricrank,
            num_prov_accept=num_prov_accept,
            extend_factor=extend_factor,
        )

    return comptable, dnode_outputs


lowvariance_highmeanmetricrank_lowkappa.__doc__ = (
    lowvariance_highmeanmetricrank_lowkappa.__doc__.format(**decision_docs)
)


def highvariance_highmeanmetricrank_highkapparatio(
    comptable,
    decision_node_idx,
    ifTrue,
    ifFalse,
    decide_comps,
    n_echos,
    n_vols=None,
    extend_factor=None,
    restrict_factor=2,
    prev_X_steps=0,
    high_perc=90,
    log_extra_report="",
    log_extra_info="",
    custom_node_label="",
    only_used_metrics=False,
):
    """
    Finds components with variance above a threshold,
    a mean metric rank above a threshold, and kappa ratio above a threshold.
    This would typically be used to identify borderline remaining components to reject.

    Parameters
    ----------
    {comptable}
    {decision_node_idx}
    {ifTrue}
    {ifFalse}
    {decide_comps}
    {n_echos}
    {n_vols}
    {extend_factor}
    {restrict_factor}
    {prev_X_steps}
    {log_extra}
    {custom_node_label}
    {only_used_metrics}

    Returns
    -------
    {basicreturns}
    """

    used_metrics = [
        "variance explained",
        "kappa",
        "rho",
        "dice_FT2",
        "signal-noise_t",
        "countsigFT2",
        "countnoise",
    ]
    if only_used_metrics:
        return used_metrics

    function_name_idx = (
        "Step {}: highvariance_highmeanmetricrank_highkapparatio".format(
            decision_node_idx
        )
    )
    if custom_node_label:
        node_label = custom_node_label
    else:
        node_label = "highvar highmeanmetricrank highkapparatio"

    if log_extra_info:
        LGR.info(log_extra_info)
    if log_extra_report:
        RepLGR.info(log_extra_report)
    metrics_exist, missing_metrics = confirm_metrics_exist(
        comptable, used_metrics, function_name=function_name_idx
    )

    comps2use = selectcomps2use(comptable, decide_comps)
    if comps2use is None:
        log_decision_tree_step(function_name_idx, comps2use, decide_comps=decide_comps)
        numTrue = 0
        numFalse = 0
        dnode_outputs = create_dnode_outputs(
            decision_node_idx, used_metrics, node_label, 0, 0
        )
    else:
        # This will either identify a previously calculated revised meanmetricrank and
        # return it or it will calculate a revised meanmetricrank, return it,
        # and add a new column to comptable that contains this new metric
        meanmetricrank, comptable = get_new_meanmetricrank(
            comptable, comps2use, decision_node_idx
        )

        # Identify components that were either provionsally accepted
        # or don't have a final classificaiton in prev_X_steps previous nodes
        previous_comps2use = prev_classified_comps(
            comptable,
            decision_node_idx,
            ["provisionalaccept", "provisionalreject", "unclassified"],
            prev_X_steps=prev_X_steps,
        )
        previous_provaccept_comps2use = prev_classified_comps(
            comptable,
            decision_node_idx,
            ["provisionalaccept"],
            prev_X_steps=prev_X_steps,
        )

        kappa_elbow = kappa_elbow_kundu(comptable, n_echos)

        # This should be the same as the MEICA 2.7 code except that I'm using provisionalaccept
        # instead of >kappa_eblow and <rho_elbow, which si how provisionally accepted components
        # are initially classified
        num_acc_guess = int(
            np.mean(
                [
                    len(previous_provaccept_comps2use),
                    np.sum(comptable.loc[previous_comps2use, "kappa"] > kappa_elbow),
                ]
            )
        )

        # a scaling factor that is either based on the number of volumes or can be
        # directly assigned
        extend_factor = get_extend_factor(n_vols=n_vols, extend_factor=extend_factor)

        varex_upper_thresh = scoreatpercentile(
            comptable.loc[previous_provaccept_comps2use, "variance explained"],
            high_perc,
        )

        # get kappa ratio
        acc_prov = prev_classified_comps(
            comptable,
            decision_node_idx,
            ["provisionalaccept"],
            prev_X_steps=prev_X_steps,
        )
        kappa_rate = (
            np.nanmax(comptable.loc[acc_prov, "kappa"])
            - np.nanmin(comptable.loc[acc_prov, "kappa"])
        ) / (
            np.nanmax(comptable.loc[acc_prov, "variance explained"])
            - np.nanmin(comptable.loc[acc_prov, "variance explained"])
        )
        LGR.info(f"Kappa rate found to be {kappa_rate} from components " f"{acc_prov}")
        comptable["kappa ratio"] = (
            kappa_rate * comptable["variance explained"] / comptable["kappa"]
        )

        conservative_guess = num_acc_guess / restrict_factor
        db_mmrank = meanmetricrank.loc[comps2use] > conservative_guess
        db_kapparatio = comptable.loc[comps2use, "kappa ratio"] > (extend_factor * 2)
        db_var_upper = comptable.loc[comps2use, "variance explained"] > (
            varex_upper_thresh * extend_factor
        )
        decision_boolean = db_mmrank & db_kapparatio & db_var_upper

        comptable = change_comptable_classifications(
            comptable, ifTrue, ifFalse, decision_boolean, str(decision_node_idx)
        )
        numTrue = np.asarray(decision_boolean).sum()
        numFalse = np.logical_not(decision_boolean).sum()
        # print(('numTrue={}, numFalse={}, numcomps2use={}'.format(
        #        numTrue, numFalse, len(comps2use))))

        log_decision_tree_step(
            function_name_idx,
            comps2use,
            numTrue=numTrue,
            numFalse=numFalse,
            ifTrue=ifTrue,
            ifFalse=ifFalse,
        )

        dnode_outputs = create_dnode_outputs(
            decision_node_idx,
            used_metrics,
            node_label,
            numTrue,
            numFalse,
            n_echos=n_echos,
            n_vols=n_vols,
            varex_threshold=varex_upper_thresh,
            restrict_factor=2,
            prev_X_steps=prev_X_steps,
            max_good_meanmetricrank=conservative_guess,
            num_acc_guess=num_acc_guess,
            extend_factor=extend_factor,
        )

    return comptable, dnode_outputs


highvariance_highmeanmetricrank_highkapparatio.__doc__ = (
    highvariance_highmeanmetricrank_highkapparatio.__doc__.format(**decision_docs)
)


def highvariance_highmeanmetricrank(
    comptable,
    decision_node_idx,
    ifTrue,
    ifFalse,
    decide_comps,
    n_echos,
    n_vols=None,
    low_perc=25,
    high_perc=90,
    extend_factor=None,
    prev_X_steps=0,
    recalc_varex_lower_thresh=False,
    log_extra_report="",
    log_extra_info="",
    custom_node_label="",
    only_used_metrics=False,
):
    """
    Finds components with variance above a threshold,
    a mean metric rank above a threshold, and kappa ratio above a threshold.
    This would typically be used to identify borderline remaining components to reject.

    Parameters
    ----------
    {comptable}
    {decision_node_idx}
    {ifTrue}
    {ifFalse}
    {decide_comps}
    {n_echos}
    {n_vols}
    {extend_factor}
    {prev_X_steps}
    {log_extra}
    {custom_node_label}
    {only_used_metrics}

    Returns
    -------
    {basicreturns}
    """

    used_metrics = [
        "variance explained",
        "kappa",
        "rho",
        "dice_FT2",
        "signal-noise_t",
        "countsigFT2",
        "countnoise",
    ]
    if only_used_metrics:
        return used_metrics

    function_name_idx = (
        "Step {}: highvariance_highmeanmetricrank_highkapparatio".format(
            (decision_node_idx)
        )
    )
    if custom_node_label:
        node_label = custom_node_label
    else:
        node_label = "highvar highmeanmetricrank highkapparatio"

    if log_extra_info:
        LGR.info(log_extra_info)
    if log_extra_report:
        RepLGR.info(log_extra_report)
    metrics_exist, missing_metrics = confirm_metrics_exist(
        comptable, used_metrics, function_name=function_name_idx
    )

    comps2use = selectcomps2use(comptable, decide_comps)
    if comps2use is None:
        log_decision_tree_step(function_name_idx, comps2use, decide_comps=decide_comps)
        numTrue = 0
        numFalse = 0
        dnode_outputs = create_dnode_outputs(
            decision_node_idx, used_metrics, node_label, 0, 0
        )
    else:
        # This will either identify a previously calculated revised meanmetricrank and
        # return it or it will calculate a revised meanmetricrank, return it,
        # and add a new column to comptable that contains this new metric
        meanmetricrank, comptable = get_new_meanmetricrank(
            comptable, comps2use, decision_node_idx
        )

        # Identify components that were either provionsally accepted
        # or don't have a final classificaiton in prev_X_steps previous nodes
        previous_comps2use = prev_classified_comps(
            comptable,
            decision_node_idx,
            ["provisionalaccept", "provisionalreject", "unclassified"],
            prev_X_steps=prev_X_steps,
        )
        previous_provaccept_comps2use = prev_classified_comps(
            comptable,
            decision_node_idx,
            ["provisionalaccept"],
            prev_X_steps=prev_X_steps,
        )

        kappa_elbow = kappa_elbow_kundu(comptable, n_echos)

        # This should be the same as the MEICA 2.7 code except that I'm using provisionalaccept
        # instead of >kappa_eblow and <rho_elbow, which si how provisionally accepted components
        # are initially classified
        num_acc_guess = int(
            np.mean(
                len(previous_provaccept_comps2use),
                np.sum(comptable.loc[previous_comps2use, "kappa"] > kappa_elbow),
            )
        )

        # a scaling factor that is either based on the number of volumes or can be
        # directly assigned
        extend_factor = get_extend_factor(n_vols=n_vols, extend_factor=extend_factor)

        conservative_guess2 = num_acc_guess * high_perc / 100.0
        db_mmrank = meanmetricrank.loc[comps2use] > conservative_guess2

        if recalc_varex_lower_thresh:
            # Note: In MEICA v2.7 code, the included components are:
            # [comps2use[:num_acc_guess]]. That would only make sense if the
            # components were sorted by variance and I don't think they were.
            # even still, this would be the the same as shifting the percentile
            # based on num_acc_guess. The threshold without num_acc_guess seems
            # equally arbitry so just keeping that for simplicity.
            varex_lower_thresh = scoreatpercentile(
                comptable.loc[comps2use, "variance explained"], low_perc
            )
        else:
            varex_lower_thresh = scoreatpercentile(
                comptable.loc[previous_provaccept_comps2use, "variance explained"],
                low_perc,
            )

        db_var_lower = comptable.loc[comps2use, "variance explained"] > (
            varex_lower_thresh * extend_factor
        )

        decision_boolean = db_mmrank & db_var_lower

        comptable = change_comptable_classifications(
            comptable, ifTrue, ifFalse, decision_boolean, str(decision_node_idx)
        )
        numTrue = np.asarray(decision_boolean).sum()
        numFalse = np.logical_not(decision_boolean).sum()
        # print(('numTrue={}, numFalse={}, numcomps2use={}'.format(
        #        numTrue, numFalse, len(comps2use))))

        log_decision_tree_step(
            function_name_idx,
            comps2use,
            numTrue=numTrue,
            numFalse=numFalse,
            ifTrue=ifTrue,
            ifFalse=ifFalse,
        )

        dnode_outputs = create_dnode_outputs(
            decision_node_idx,
            used_metrics,
            node_label,
            numTrue,
            numFalse,
            n_echos=n_echos,
            n_vols=n_vols,
            varex_threshold=varex_lower_thresh,
            prev_X_steps=prev_X_steps,
            max_good_meanmetricrank=conservative_guess2,
            num_acc_guess=num_acc_guess,
            extend_factor=extend_factor,
        )

    return comptable, dnode_outputs


highvariance_highmeanmetricrank.__doc__ = (
    highvariance_highmeanmetricrank.__doc__.format(**decision_docs)
)


def highvariance_lowkappa(
    comptable,
    decision_node_idx,
    ifTrue,
    ifFalse,
    decide_comps,
    n_echos,
    low_perc=25,
    log_extra_report="",
    log_extra_info="",
    custom_node_label="",
    only_used_metrics=False,
):
    """
    Finds components with variance above a threshold,
    a mean metric rank above a threshold, and kappa ratio above a threshold.
    This would typically be used to identify borderline remaining components to reject.

    Parameters
    ----------
    {comptable}
    {decision_node_idx}
    {ifTrue}
    {ifFalse}
    {decide_comps}
    {n_echos}
    {log_extra}
    {custom_node_label}
    {only_used_metrics}

    Returns
    -------
    {basicreturns}
    """

    used_metrics = ["variance explained", "kappa"]
    if only_used_metrics:
        return used_metrics

    function_name_idx = "Step {}: highvariance_lowkappa".format(decision_node_idx)
    if custom_node_label:
        node_label = custom_node_label
    else:
        node_label = "highvariance lowkappa"

    if log_extra_info:
        LGR.info(log_extra_info)
    if log_extra_report:
        RepLGR.info(log_extra_report)
    metrics_exist, missing_metrics = confirm_metrics_exist(
        comptable, used_metrics, function_name=function_name_idx
    )

    comps2use = selectcomps2use(comptable, decide_comps)
    if comps2use is None:
        log_decision_tree_step(function_name_idx, comps2use, decide_comps=decide_comps)
        numTrue = 0
        numFalse = 0
        dnode_outputs = create_dnode_outputs(
            decision_node_idx, used_metrics, node_label, 0, 0
        )
    else:
        kappa_elbow = kappa_elbow_kundu(comptable, n_echos)
        db_kappa = comptable.loc[comps2use, "kappa"] <= kappa_elbow

        # Note: In MEICA v2.7 code, the included components are:
        # [comps2use[:num_acc_guess]]. That would only make sense if the
        # components were sorted by variance and I don't think they were.
        # even still, this would be the the same as shifting the percentile
        # based on num_acc_guess. The threshold without num_acc_guess seems
        # equally arbitry so just keeping that for simplicity.
        varex_lower_thresh = scoreatpercentile(
            comptable.loc[comps2use, "variance explained"], low_perc
        )

        db_var_lower = (
            comptable.loc[comps2use, "variance explained"] > varex_lower_thresh
        )

        decision_boolean = db_kappa & db_var_lower

        comptable = change_comptable_classifications(
            comptable, ifTrue, ifFalse, decision_boolean, str(decision_node_idx)
        )
        numTrue = np.asarray(decision_boolean).sum()
        numFalse = np.logical_not(decision_boolean).sum()
        # print(('numTrue={}, numFalse={}, numcomps2use={}'.format(
        #        numTrue, numFalse, len(comps2use))))

        log_decision_tree_step(
            function_name_idx,
            comps2use,
            numTrue=numTrue,
            numFalse=numFalse,
            ifTrue=ifTrue,
            ifFalse=ifFalse,
        )

        dnode_outputs = create_dnode_outputs(
            decision_node_idx,
            used_metrics,
            node_label,
            numTrue,
            numFalse,
            n_echos=n_echos,
            varex_threshold=varex_lower_thresh,
        )

    return comptable, dnode_outputs


highvariance_lowkappa.__doc__ = highvariance_lowkappa.__doc__.format(**decision_docs)
