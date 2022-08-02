"""
Functions that will be used as steps in a decision tree
"""
import logging
import numpy as np
import pandas as pd

# from scipy import stats

from scipy.stats import scoreatpercentile
from tedana.stats import getfbounds
from tedana.selection.selection_utils import (
    confirm_metrics_exist,
    selectcomps2use,
    log_decision_tree_step,
    change_comptable_classifications,
    getelbow,
    get_extend_factor,
    kappa_elbow_kundu,
    # get_new_meanmetricrank,
)

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")
RefLGR = logging.getLogger("REFERENCES")

decision_docs = {
    "selector": """\
selector: :obj:`tedana.selection.ComponentSelector`
    This structure contains most of the information needed to execute each
    decision node function and to store the ouput of the function. The class
    description has full details. Key elements include: component_table:
    The metrics for each component, and the classification
    labels and tags; cross_component_metrics: Values like the kappa and rho
    elbows that are used to create decision criteria; nodes: Information on
    the function calls for each step in the decision tree; and
    current_node_idx: which is the ordered index for when a function is
    called in the decision tree\
""",
    "ifTrueFalse": """\
ifTrue, ifFalse: :obj:`str`
    If the condition in this step is true or false, give the component
    the label in this string. Options are 'accepted', 'rejected',
    'nochange', or intermediate_classification labels predefined in the
    decision tree. If 'nochange' then don't change the current component
    classification\
""",
    "decide_comps": """\
decide_comps: :obj:`str` or :obj:`list[str]`
    This is string or a list of strings describing what classifications
    of components to operate on, using default or intermediate_classification
    labels. For example: decide_comps='unclassified' means to operate only on
    unclassified components. The label 'all' will operate on all components
    regardess of classification.\
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
    "tag_ifTrueFalse": """\
tag_ifTrue, tag_ifFalse: :obj:`str`
    A string containing a label in classification_tags that will be added to
    the classification_tags column in component_table if a component is
    classified as true or false. default=None
""",
    "basicreturns": """\
selector: :obj:`tedana.selection.ComponentSelector`
    The key fields that will be changed in selector are the component
    classifications and tags in component_table or a new metric that is
    added to cross_component_metrics. The output field for the current
    node will also be updated to include relevant information including
    the use_metrics of the node, and the numTrue and numFalse components
    the call to the node's function.\
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
}


def manual_classify(
    selector,
    decide_comps,
    new_classification,
    clear_classification_tags=False,
    log_extra_report="",
    log_extra_info="",
    custom_node_label="",
    only_used_metrics=False,
    tag=None,
    dont_warn_reclassify=False,
):
    """
    Explicitly assign a classifictation, defined in new_classification,
    to all the components in decide_comps.

    Parameters
    ----------
    {selector}
    {decide_comps}
    new_classification: :obj: `str`
        Assign all components identified in decide_comps the classification
        in new_classification. Options are 'unclassified', 'accepted',
        'rejected', or intermediate_classification labels predefined in the
        decision tree
    clear_classification_tags: :obj: `bool`
        If True, reset all values in the 'classification_tags' column to empty
        strings. This also can create the classification_tags column if it
        does not already exist
        If False, do nothing.
    tag: :obj: `str`
        A classification tag to assign to all components being reclassified.
        This should be one of the tags defined by classification_tags in
        the decision tree specification
    dont_warn_reclassify: :obj:`bool`
        By default, if this function changes a component classification from accepted or
        rejected to something else, it gives a warning, since those should be terminal
        classifications. If this is True, that warning is suppressed.
        (Useful if manual_classify is used to reset all labels to unclassified).
        default=False
    {log_extra}
    {custom_node_label}
    {only_used_metrics}


    Returns
    -------
    {basicreturns}

    Note
    ----
    This was designed with three use
    cases in mind:
    1. Set the classifications of all components to unclassified for the first
    node of a decision tree. clear_classification_tags=True is recommended for
    this use case
    2. Shift all components between classifications, such as provisionalaccept
    to accepted for the penultimate node in the decision tree.
    3. Manually re-classify components by number based on user observations.

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

    comps2use = selectcomps2use(selector, decide_comps)

    if not comps2use:
        log_decision_tree_step(function_name_idx, comps2use, decide_comps=decide_comps)
        outputs["numTrue"] = 0
        outputs["numFalse"] = 0
    else:
        decision_boolean = pd.Series(True, index=comps2use)
        selector, outputs["numTrue"], outputs["numFalse"] = change_comptable_classifications(
            selector,
            ifTrue,
            ifFalse,
            decision_boolean,
            tag_ifTrue=tag,
            dont_warn_reclassify=dont_warn_reclassify,
        )
        # outputs["numTrue"] = decision_boolean.sum()
        # outputs["numFalse"] = np.logical_not(decision_boolean).sum()

        log_decision_tree_step(
            function_name_idx,
            comps2use,
            numTrue=outputs["numTrue"],
            numFalse=outputs["numFalse"],
            ifTrue=ifTrue,
            ifFalse=ifFalse,
        )

    if clear_classification_tags:
        selector.component_table["classification_tags"] = ""
        LGR.info(function_name_idx + " component classification tags are cleared")

    selector.tree["nodes"][selector.current_node_idx]["outputs"] = outputs

    return selector


manual_classify.__doc__ = manual_classify.__doc__.format(**decision_docs)


def dec_left_op_right(
    selector,
    ifTrue,
    ifFalse,
    decide_comps,
    op,
    left,
    right,
    left_scale=1,
    right_scale=1,
    op2=None,
    left2=None,
    right2=None,
    left2_scale=1,
    right2_scale=1,
    op3=None,
    left3=None,
    right3=None,
    left3_scale=1,
    right3_scale=1,
    log_extra_report="",
    log_extra_info="",
    custom_node_label="",
    only_used_metrics=False,
    tag_ifTrue=None,
    tag_ifFalse=None,
):
    """
    Tests a relationship between (left_scale*)left and (right_scale*right)
    using an operator, like >, defined with op
    This can be used to directly compare any 2 metrics and use that info
    to change component classification. If either metric is a number,
    this can also compare a metric against a fixed threshold.

    Parameters
    ----------
    {selector}
    {ifTrueFalse}
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
        kappa>(2*rho). These also be a string that labels a value in
        cross_component_metrics, since those will resolve to a single value.
        This cannot be a label for a component_table column since that would
        output a different value for each component. default=1
    op2: :ojb:`str`, optional
    left2, right2, left3, right3: :obj:`str` or :obj:`float`, optional
    left2_scale, right2_scale, left3_scale, right3_scale: :obj:`float`, optional
        This function can also be used to calculate the intersection of two or three
        boolean statements. If op2, left2, and right2 are defined then
        this function returns
        (left_scale*)left op (right_scale*right) AND (left2_scale*)left2 op2 (right2_scale*right2)
        if the "3" parameters are also defined then it's the intersection of all 3 statements
    {log_extra}
    {custom_node_label}
    {only_used_metrics}
    {tag_ifTrueFalse}

    Returns
    -------
    {basicreturns}

    Note
    ----
    This function is ideally run with one boolean statement at a time so that
    the result of each boolean is logged. For example, it's better to test
    kappa>kappa_elbow and rho>rho_elbow with two separate calls to this function
    so that the results of each can be easily viewed. That said, particularly for
    the original kundu decision tree, if you're making decisions on components with
    various classifications based on multiple boolean statements, the decesion tree
    becomes really messy and the added functionality here is useful.
    Combinations of boolean statements only test with "and" and not "or". This is
    an intentional decision because, if a classification changes if A or B are true
    then the results of each should be logged separately
    """

    # predefine all outputs that should be logged
    outputs = {
        "decision_node_idx": selector.current_node_idx,
        "used_metrics": set(),
        "used_cross_component_metrics": set(),
        "node_label": None,
        "numTrue": None,
        "numFalse": None,
    }

    function_name_idx = f"Step {selector.current_node_idx}: left_op_right"
    # Only select components if the decision tree is being run
    if not only_used_metrics:
        comps2use = selectcomps2use(selector, decide_comps)

    def identify_used_metric(val, isnum=False):
        """
        Parse the left or right values or scalers to see if they are an
        existing used_metric or cross_component_metric
        If the value already a number, no parse would be needed

        This is also used on left_scale and right_scale to convert
        a value in cross_component_metrics to a number. Set the isnum
        flag to true for those inputs and this will raise an error
        if a number isn't loaded
        """
        orig_val = val
        if isinstance(val, str):
            if val in selector.component_table.columns:
                outputs["used_metrics"].update([val])
            elif val in selector.cross_component_metrics:
                outputs["used_cross_component_metrics"].update([val])
                val = selector.cross_component_metrics[val]
            # If decision tree is being run, then throw errors or messages
            #  if a component doesn't exist. If this is just getting a list
            #  of metrics to be used, then don't bring up warnings
            elif not only_used_metrics:
                if not comps2use:
                    LGR.info(
                        f"{function_name_idx}: {val} is neither a metric in "
                        "selector.component_table nor selector.cross_component_metrics, "
                        f"but no components with {decide_comps} remain by this node "
                        "so nothing happens"
                    )
                else:
                    raise ValueError(
                        f"{val} is neither a metric in selector.component_table nor selector.cross_component_metrics"
                    )
        if isnum:
            if not isinstance(val, (int, float)):
                raise ValueError(f"{orig_val} must be a number. It is {val}")
        return val

    legal_ops = (">", ">=", "==", "<=", "<")

    def confirm_valid_conditional(left_scale, left_val, right_scale, right_val, op_val):
        """
        Makes sure the left_scale, left_val, right_scale, right_val, and
        operator variables combine into a valid conditional statement
        """

        left_val = identify_used_metric(left_val)
        right_val = identify_used_metric(right_val)
        left_scale = identify_used_metric(left_scale, isnum=True)
        right_scale = identify_used_metric(right_scale, isnum=True)

        if op_val not in legal_ops:
            raise ValueError(f"{op_val} is not a binary comparison operator, like > or <")
        return left_scale, left_val, right_scale, right_val

    def operator_scale_descript(val_scale, val):
        """
        Return a string with one element from the mathematical expression
        If val_scale is not 1, include scaling factor (rounded to 2 decimals)
        If val is a column in the component_table output the column label
        If val is a number (either an inputted number or from cross_component_metrics
        include the number (rounded to 2 decimals)
        This output is used to great a descriptor for visualizing the decision tree
        Unrounded values are saved and rounding here will not affect results
        """
        if not isinstance(val, str):
            val = str(round(val, 2))
        if val_scale == 1:
            return val
        else:
            return f"{round(val_scale,2)}*{val}"

    left_scale, left, right_scale, right = confirm_valid_conditional(
        left_scale, left, right_scale, right, op
    )
    descript_left = operator_scale_descript(left_scale, left)
    descript_right = operator_scale_descript(right_scale, right)
    is_compound = 0

    # If any of the values for the second boolean statement are set
    if left2 or right2 or op2:
        # Check if they're all set & use them all or raise an error
        if left2 and right2 and op2:
            is_compound = 2
            left2_scale, left2, right2_scale, right2 = confirm_valid_conditional(
                left2_scale, left2, right2_scale, right2, op2
            )
            descript_left2 = operator_scale_descript(left2_scale, left2)
            descript_right2 = operator_scale_descript(right2_scale, right2)
        else:
            raise ValueError(
                "left_op_right can check if a first and second boolean "
                "statement are both true. This call includes some but not "
                "all variables to define the second boolean statement "
                f"left2={left2}, right2={right2}, op2={op2}"
            )

    # If any of the values for the second boolean statement are set
    if left3 or right3 or op3:
        if is_compound == 0:
            raise ValueError(
                "left_op_right is includes parameters for a third conditional "
                "(left3, right3, or op3) statement without setting the "
                "second statement"
            )
        # Check if they're all set & use them all or raise an error
        if left3 and right3 and op3:
            is_compound = 3
            left3_scale, left3, right3_scale, right3 = confirm_valid_conditional(
                left3_scale, left3, right3_scale, right3, op3
            )
            descript_left3 = operator_scale_descript(left3_scale, left3)
            descript_right3 = operator_scale_descript(right3_scale, right3)
        else:
            raise ValueError(
                "left_op_right can check if three boolean "
                "statements are all true. This call includes some but not "
                "all variables to define the third boolean statement "
                f"left3={left3}, right3={right3}, op3={op3}"
            )

    if only_used_metrics:
        return outputs["used_metrics"]

    if custom_node_label:
        outputs["node_label"] = custom_node_label
    elif is_compound == 0:
        outputs["node_label"] = f"{descript_left}{op}{descript_right}"
    elif is_compound == 2:
        outputs["node_label"] = [
            f"{descript_left}{op}{descript_right} & " f"{descript_left2}{op2}{descript_right2}"
        ]
    elif is_compound == 3:
        outputs["node_label"] = [
            f"{descript_left}{op}{descript_right} & "
            f"{descript_left2}{op2}{descript_right2} & "
            f"{descript_left3}{op3}{descript_right3}"
        ]

    # Might want to add additional default logging to functions here
    # The function input will be logged before the function call
    if log_extra_info:
        LGR.info(log_extra_info)
    if log_extra_report:
        RepLGR.info(log_extra_report)

    confirm_metrics_exist(
        selector.component_table, outputs["used_metrics"], function_name=function_name_idx
    )

    def parse_vals(val):
        """Get the actual metric values for the selected components or return the constant int or float"""
        if isinstance(val, str):
            return selector.component_table.loc[comps2use, val].copy()
        else:
            return val  # should be a fixed number

    if not comps2use:
        outputs["numTrue"] = 0
        outputs["numFalse"] = 0
        log_decision_tree_step(
            function_name_idx,
            comps2use,
            decide_comps=decide_comps,
            ifTrue=outputs["numTrue"],
            ifFalse=outputs["numFalse"],
        )

    else:
        left1_val = parse_vals(left)
        right1_val = parse_vals(right)
        decision_boolean = eval(f"(left_scale*left1_val) {op} (right_scale * right1_val)")
        if is_compound >= 2:
            left2_val = parse_vals(left2)
            right2_val = parse_vals(right2)
            statement1 = decision_boolean.copy()
            statement2 = eval(f"(left2_scale*left2_val) {op2} (right2_scale * right2_val)")
            # logical dot product for compound statement
            decision_boolean = statement1 * statement2
        if is_compound == 3:
            left3_val = parse_vals(left3)
            right3_val = parse_vals(right3)
            # statement 1 is now the combination of the first two conditional statements
            statement1 = decision_boolean.copy()
            # statement 2 is now the third conditional statement
            statement2 = eval(f"(left3_scale*left3_val) {op2} (right3_scale * right3_val)")
            # logical dot product for compound statement
            decision_boolean = statement1 * statement2

        (selector, outputs["numTrue"], outputs["numFalse"],) = change_comptable_classifications(
            selector,
            ifTrue,
            ifFalse,
            decision_boolean,
            tag_ifTrue=tag_ifTrue,
            tag_ifFalse=tag_ifFalse,
        )
        # outputs["numTrue"] = np.asarray(decision_boolean).sum()
        # outputs["numFalse"] = np.logical_not(decision_boolean).sum()

        log_decision_tree_step(
            function_name_idx,
            comps2use,
            numTrue=outputs["numTrue"],
            numFalse=outputs["numFalse"],
            ifTrue=ifTrue,
            ifFalse=ifFalse,
        )

    selector.tree["nodes"][selector.current_node_idx]["outputs"] = outputs

    return selector


dec_left_op_right.__doc__ = dec_left_op_right.__doc__.format(**decision_docs)


def dec_variance_lessthan_thresholds(
    selector,
    ifTrue,
    ifFalse,
    decide_comps,
    var_metric="variance explained",
    single_comp_threshold=0.1,
    all_comp_threshold=1.0,
    log_extra_report="",
    log_extra_info="",
    custom_node_label="",
    only_used_metrics=False,
    tag_ifTrue=None,
    tag_ifFalse=None,
):
    """
    Finds components with variance<single_comp_threshold.
    If the sum of the variance for all components that meet this criteria
    is greater than all_comp_threshold then keep the lowest variance
    components so that the sum of their variances is less than all_comp_threshold

    Parameters
    ----------
    {selector}
    {ifTrueFalse}
    {decide_comps}
    var_metric: :obj:`str`
        The name of the metric in comptable for variance. default=varexp
        This is an option so that it is possible to set this to normvarexp
        or some other variance measure
    single_comp_threshold: :obj:`float`
        The threshold for which all components need to have lower variance.
        default=0.1
    all_comp_threshold: :obj: `float`
        The threshold for which the sum of all components<single_comp_threshold
        needs to be under. default=1.0
    {log_extra}
    {custom_node_label}
    {only_used_metrics}
    {tag_ifTrueFalse}

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

    function_name_idx = "Step {}: variance_lt_thresholds".format(selector.current_node_idx)
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

    comps2use = selectcomps2use(selector, decide_comps)
    confirm_metrics_exist(
        selector.component_table, outputs["used_metrics"], function_name=function_name_idx
    )

    if not comps2use:
        outputs["numTrue"] = 0
        outputs["numFalse"] = 0
        log_decision_tree_step(
            function_name_idx,
            comps2use,
            decide_comps=decide_comps,
            ifTrue=outputs["numTrue"],
            ifFalse=outputs["numFalse"],
        )
    else:
        variance = selector.component_table.loc[comps2use, var_metric]
        decision_boolean = variance < single_comp_threshold
        # if all the low variance components sum above all_comp_threshold
        # keep removing the highest remaining variance component until
        # the sum is below all_comp_threshold. This is an inefficient
        # way to do this, but it works & should never cause an infinite loop
        if variance[decision_boolean].sum() > all_comp_threshold:
            while variance[decision_boolean].sum() > all_comp_threshold:
                tmpmax = variance == variance[decision_boolean].max()
                decision_boolean[tmpmax] = False
        (selector, outputs["numTrue"], outputs["numFalse"],) = change_comptable_classifications(
            selector,
            ifTrue,
            ifFalse,
            decision_boolean,
            tag_ifTrue=tag_ifTrue,
            tag_ifFalse=tag_ifFalse,
        )
        # outputs["numTrue"] = np.asarray(decision_boolean).sum()
        # outputs["numFalse"] = np.logical_not(decision_boolean).sum()

        log_decision_tree_step(
            function_name_idx,
            comps2use,
            numTrue=outputs["numTrue"],
            numFalse=outputs["numFalse"],
            ifTrue=ifTrue,
            ifFalse=ifFalse,
        )

    selector.tree["nodes"][selector.current_node_idx]["outputs"] = outputs
    return selector


dec_variance_lessthan_thresholds.__doc__ = dec_variance_lessthan_thresholds.__doc__.format(
    **decision_docs
)


def calc_kappa_rho_elbows_kundu(
    selector,
    decide_comps,
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
    {selector}
    {decide_comps}
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

    Note
    ----
    This script is currently hard coded for a specific way to calculate kappa and rho elbows
    based on the method by Kundu in the MEICA v2.7 code. Another elbow calculation would
    require a distinct function. Ideally, there can be one elbow function can allows for
    some more flexible options

    This also uses all unclassified components as part of the elbow calculation, irregardless
    of what is in decide_comps.
    """

    # If kappa_only or rho_only is true kappa or rho might not actually be
    # used, but, as of now, both are required to run this function

    outputs = {
        "decision_node_idx": selector.current_node_idx,
        "node_label": None,
        "n_echos": selector.n_echos,
        "varex_upper_p": None,
    }
    if not (kappa_only ^ rho_only):
        # if neither or both kappa and rho_only are set
        outputs["used_metrics"] = set(["kappa", "rho"])
        outputs["calc_cross_comp_metrics"] = [
            "kappa_elbow_kundu",
            "rho_elbow_kundu",
            "varex_upper_p",
        ]
        outputs["kappa_elbow_kundu"] = None
        outputs["rho_elbow_kundu"] = None
        calc_kappa = True
        calc_rho = True
    elif kappa_only:
        outputs["used_metrics"] = set(["kappa"])
        outputs["calc_cross_comp_metrics"] = [
            "kappa_elbow_kundu",
            "varex_upper_p",
        ]
        outputs["kappa_elbow_kundu"] = None
        calc_kappa = True
        calc_rho = False
    elif rho_only:
        outputs["used_metrics"] = set(["rho"])
        outputs["calc_cross_comp_metrics"] = [
            "rho_elbow_kundu",
            "varex_upper_p",
        ]
        outputs["rho_elbow_kundu"] = None
        calc_kappa = False
        calc_rho = True

    if only_used_metrics:
        return outputs["used_metrics"]

    function_name_idx = f"Step {selector.current_node_idx}: calc_kappa_rho_elbows_kundu"

    if ("kappa_elbow_kundu" in selector.cross_component_metrics) and (
        "kappa_elbow_kundu" in outputs["calc_cross_comp_metrics"]
    ):
        LGR.warning(
            f"kappa_elbow_kundu already calculated. Overwriting previous value in {function_name_idx}"
        )
    if ("rho_elbow_kundu" in selector.cross_component_metrics) and (
        "rho_elbow_kundu" in outputs["calc_cross_comp_metrics"]
    ):
        LGR.warning(
            f"rho_elbow_kundu already calculated. Overwriting previous value in {function_name_idx}"
        )
    if "varex_upper_p" in selector.cross_component_metrics:
        LGR.warning(
            f"varex_upper_p already calculated. Overwriting previous value in {function_name_idx}"
        )

    if custom_node_label:
        outputs["node_label"] = custom_node_label
    else:
        if not (kappa_only ^ rho_only):
            outputs["node_label"] = "Calc Kappa & Rho Elbows"
        elif kappa_only:
            outputs["node_label"] = "Calc Kappa Elbow"
        elif rho_only:
            outputs["node_label"] = "Calc Rho Elbow"

    LGR.info(
        "Note: This matches the elbow selecton criteria in Kundu's MEICA v2.7"
        " except there is a variance threshold that is used for the rho criteria that "
        "really didn't make sense and is being excluded."
    )

    if log_extra_info:
        LGR.info(log_extra_info)
    if log_extra_report:
        RepLGR.info(log_extra_report)

    comps2use = selectcomps2use(selector, decide_comps)
    confirm_metrics_exist(
        selector.component_table, outputs["used_metrics"], function_name=function_name_idx
    )

    unclassified_comps2use = selectcomps2use(selector, "unclassified")

    if (not comps2use) or (not unclassified_comps2use):
        if not comps2use:
            # outputs["numTrue"] = 0
            # outputs["numFalse"] = 0
            log_decision_tree_step(
                function_name_idx,
                comps2use,
                decide_comps=decide_comps,
                # ifTrue=outputs["numTrue"],
                # ifFalse=outputs["numFalse"],
            )
        if not unclassified_comps2use:
            # outputs["numTrue"] = 0
            # outputs["numFalse"] = 0
            log_decision_tree_step(
                function_name_idx,
                comps2use,
                decide_comps="unclassified",
                # ifTrue=outputs["numTrue"],
                # ifFalse=outputs["numFalse"],
            )
    else:
        if calc_kappa:
            outputs["kappa_elbow_kundu"] = kappa_elbow_kundu(
                selector.component_table, selector.n_echos
            )
            selector.cross_component_metrics["kappa_elbow_kundu"] = outputs["kappa_elbow_kundu"]

        # The first elbow used to be for rho values of the unclassified components
        # excluding a few based on differences of variance. Now it's all unclassified
        # components
        # Upper limit for variance explained is median across components with high
        # Kappa values. High Kappa is defined as Kappa above Kappa elbow.
        f05, _, f01 = getfbounds(selector.n_echos)
        outputs["varex_upper_p"] = np.median(
            selector.component_table.loc[
                selector.component_table["kappa"]
                > getelbow(selector.component_table["kappa"], return_val=True),
                "variance explained",
            ]
        )
        selector.cross_component_metrics["varex_upper_p"] = outputs["varex_upper_p"]

        ncls = unclassified_comps2use.copy()
        for i_loop in range(3):
            temp_comptable = selector.component_table.loc[ncls].sort_values(
                by=["variance explained"], ascending=False
            )
            diff_vals = temp_comptable["variance explained"].diff(-1)
            diff_vals = diff_vals.fillna(0)
            ncls = temp_comptable.loc[diff_vals < outputs["varex_upper_p"]].index.values
        # kappa_elbow was already calculated in kappa_elbow_kundu above
        # kappas_nonsig = comptable.loc[comptable["kappa"] < f01, "kappa"]
        # kappa_elbow = np.min(
        #     (
        #         getelbow(kappas_nonsig, return_val=True),
        #         getelbow(comptable["kappa"], return_val=True),
        #     )
        # )
        if calc_rho:
            outputs["rho_elbow_kundu"] = np.mean(
                (
                    getelbow(selector.component_table.loc[ncls, "rho"], return_val=True),
                    getelbow(selector.component_table["rho"], return_val=True),
                    f05,
                )
            )
            selector.cross_component_metrics["rho_elbow_kundu"] = outputs["rho_elbow_kundu"]

        # print(('numTrue={}, numFalse={}, numcomps2use={}'.format(
        #        numTrue, numFalse, len(comps2use))))
        log_decision_tree_step(function_name_idx, comps2use, calc_outputs=outputs)

    selector.tree["nodes"][selector.current_node_idx]["outputs"] = outputs

    return selector


calc_kappa_rho_elbows_kundu.__doc__ = calc_kappa_rho_elbows_kundu.__doc__.format(**decision_docs)


def dec_classification_doesnt_exist(
    selector,
    new_classification,
    decide_comps,
    class_comp_exists,
    log_extra_report="",
    log_extra_info="",
    custom_node_label="",
    only_used_metrics=False,
    tag_ifTrue=None,
):
    """
    If there are no compontents with a classification specified in class_comp_exists,
    change the classification of all components in decide_comps
    Parameters
    ----------
    {selector}
    new_classification: :obj: `str`
        Assign all components identified in decide_comps the classification
        in new_classification. Options are 'unclassified', 'accepted',
        'rejected', or intermediate_classification labels predefined in the
        decision tree
    {decide_comps}
    class_comp_exists: :obj:`str` or :obj:`list[str]` or :obj:`int` or :obj:`list[int]`
        This has the same structure options as decide_comps. This function tests
        whether any components have the classifications defined in this variable.
    {log_extra}
    {custom_node_label}
    {only_used_metrics}
    {tag_ifTrueFalse}


    Returns
    -------
    {basicreturns}

    """

    # predefine all outputs that should be logged
    outputs = {
        "decision_node_idx": selector.current_node_idx,
        "used_metrics": set(),
        "used_cross_component_metrics": set(),
        "node_label": None,
        "numTrue": None,
        "numFalse": None,
    }

    if only_used_metrics:
        return outputs["used_metrics"]

    function_name_idx = "Step {}: classification_doesnt_exist".format((selector.current_node_idx))
    if custom_node_label:
        outputs["node_label"] = custom_node_label
    else:
        outputs["node_label"] = f"Change {decide_comps} if {class_comp_exists} doesn't exist"

    if log_extra_info:
        LGR.info(log_extra_info)
    if log_extra_report:
        RepLGR.info(log_extra_report)

    ifTrue = new_classification
    ifFalse = "nochange"

    comps2use = selectcomps2use(selector, decide_comps)

    do_comps_exist = selectcomps2use(selector, class_comp_exists)

    if (not comps2use) or (do_comps_exist):
        outputs["numTrue"] = 0
        # If nothing chanages, then assign the number of components in comps2use to numFalse
        outputs["numFalse"] = len(comps2use)
        log_decision_tree_step(
            function_name_idx,
            comps2use,
            decide_comps=decide_comps,
            ifTrue=outputs["numTrue"],
            ifFalse=outputs["numFalse"],
        )
    else:  # do_comps_exist is None:
        # should be True for all components in comps2use
        # decision_boolean = pd.Series(data=False, index=np.arange(len(selector.component_table)), dtype=bool)
        # decision_boolean.iloc[comps2use] = True
        decision_boolean = pd.Series(True, index=comps2use)

        selector, outputs["numTrue"], outputs["numFalse"] = change_comptable_classifications(
            selector,
            ifTrue,
            ifFalse,
            decision_boolean,
            tag_ifTrue=tag_ifTrue,
        )

        log_decision_tree_step(
            function_name_idx,
            comps2use,
            numTrue=outputs["numTrue"],
            numFalse=outputs["numFalse"],
            ifTrue=ifTrue,
            ifFalse=ifFalse,
        )

    selector.tree["nodes"][selector.current_node_idx]["outputs"] = outputs

    return selector


dec_classification_doesnt_exist.__doc__ = dec_classification_doesnt_exist.__doc__.format(
    **decision_docs
)


def calc_varex_thresh(
    selector,
    decide_comps,
    thresh_label,
    percentile_thresh,
    log_extra_report="",
    log_extra_info="",
    custom_node_label="",
    only_used_metrics=False,
):
    """
    Calculates the variance explained threshold to use in the kundu decision tree.
    Will save a high or low percentile threshold depending on highlow_thresh

    Parameters
    ----------
    {selector}
    {decide_comps}
    thresh_label: :obj:`str`
        The threshold will be saved in "varex_(thresh_label)_thresh"
        In the original kundu decision tree this was either "upper" or "lower"
        If passed an empty string,t hen will be saved as "varex_thresh"
    percentile_thresh: :obj:`int`
        A percentile threshold to apply to components to set the variance threshold.
        In the original kundu decision tree this was 90 for varex_upper_thresh and
        25 for varex_lower_thresh
    {log_extra}
    {custom_node_label}
    {only_used_metrics}

    Returns
    -------
    {basicreturns}

    """

    function_name_idx = f"Step {selector.current_node_idx}: calc_varex_thresh"
    thresh_label = thresh_label.lower()
    if thresh_label is None or thresh_label is "":
        varex_name = "varex_thresh"
        perc_name = "perc"
    else:
        varex_name = f"varex_{thresh_label}_thresh"
        perc_name = f"{thresh_label}_perc"

    outputs = {
        "decision_node_idx": selector.current_node_idx,
        "node_label": None,
        varex_name: None,
        "used_metrics": set(["variance explained"]),
    }
    if (
        isinstance(percentile_thresh, (int, float))
        and (percentile_thresh > 0)
        and (percentile_thresh < 100)
    ):
        outputs["calc_cross_comp_metrics"] = [varex_name, perc_name]
        outputs[perc_name] = percentile_thresh
    else:
        raise ValueError(
            f"percentile_thresh must be a number between 0 & 100. It is: {percentile_thresh}"
        )

    if only_used_metrics:
        return outputs["used_metrics"]

    if varex_name in selector.cross_component_metrics:
        LGR.warning(
            f"{varex_name} already calculated. Overwriting previous value in {function_name_idx}"
        )

    if perc_name in selector.cross_component_metrics:
        LGR.warning(
            f"{perc_name} already calculated. Overwriting previous value in {function_name_idx}"
        )

    if custom_node_label:
        outputs["node_label"] = custom_node_label
    else:
        outputs["node_label"] = f"Calc {varex_name}"

    if log_extra_info:
        LGR.info(log_extra_info)
    if log_extra_report:
        RepLGR.info(log_extra_report)

    comps2use = selectcomps2use(selector, decide_comps)
    confirm_metrics_exist(
        selector.component_table, outputs["used_metrics"], function_name=function_name_idx
    )

    if not comps2use:
        log_decision_tree_step(
            function_name_idx,
            comps2use,
            decide_comps=decide_comps,
        )
    else:

        outputs[varex_name] = scoreatpercentile(
            selector.component_table.loc[comps2use, "variance explained"], percentile_thresh
        )

        selector.cross_component_metrics[varex_name] = outputs[varex_name]

        log_decision_tree_step(function_name_idx, comps2use, calc_outputs=outputs)

    selector.tree["nodes"][selector.current_node_idx]["outputs"] = outputs

    return selector


calc_varex_thresh.__doc__ = calc_varex_thresh.__doc__.format(**decision_docs)


def calc_extend_factor(
    selector,
    log_extra_report="",
    log_extra_info="",
    custom_node_label="",
    only_used_metrics=False,
    extend_factor=None,
):
    """
    Calculates the scaler used to set a threshold for d_table_score

    Parameters
    ----------
    {selector}
    {decide_comps}
    {log_extra}
    {custom_node_label}
    {only_used_metrics}
    extend_factor: :obj:`float`
        If a number, then use rather than calculating anything. If None than calculate. default=None

    Returns
    -------
    {basicreturns}

    """

    outputs = {
        "used_metrics": set(),
        "decision_node_idx": selector.current_node_idx,
        "node_label": None,
        "extend_factor": None,
        "calc_cross_comp_metrics": ["extend_factor"],
    }

    if only_used_metrics:
        return outputs["used_metrics"]

    function_name_idx = f"Step {selector.current_node_idx}: calc_extend_factor"

    if "extend_factor" in selector.cross_component_metrics:
        LGR.warning(
            f"extend_factor already calculated. Overwriting previous value in {function_name_idx}"
        )

    if custom_node_label:
        outputs["node_label"] = custom_node_label
    else:
        outputs["node_label"] = "Calc extend_factor"

    if log_extra_info:
        LGR.info(log_extra_info)
    if log_extra_report:
        RepLGR.info(log_extra_report)

    outputs["extend_factor"] = get_extend_factor(
        n_vols=selector.cross_component_metrics["n_vols"], extend_factor=extend_factor
    )

    selector.cross_component_metrics["extend_factor"] = outputs["extend_factor"]

    log_decision_tree_step(function_name_idx, None, calc_outputs=outputs, comps_needed=False)

    selector.tree["nodes"][selector.current_node_idx]["outputs"] = outputs

    return selector


calc_extend_factor.__doc__ = calc_extend_factor.__doc__.format(**decision_docs)


def calc_max_good_meanmetricrank(
    selector,
    decide_comps,
    metric_suffix=None,
    log_extra_report="",
    log_extra_info="",
    custom_node_label="",
    only_used_metrics=False,
):
    """
    Calculates the max_good_meanmetricrank to use in the kundu decision tree
    This is the number of components seleted with decide_comps * the extend_factor
    calculated in calc_extend_factor

    Parameters
    ----------
    {selector}
    {decide_comps}
    metric_suffix: :obj:`str`
        By default, this will output a value called "max_good_meanmetricrank"
        If this variable is not None or "" then it will output: "max_good_meanmetricrank_[metric_suffix]
    {log_extra}
    {custom_node_label}
    {only_used_metrics}

    Returns
    -------
    {basicreturns}

    Note
    ----
    "meanmetricrank" is the same as "d_table_score" and is used to set a threshold for
    the "d_table" values in the component table. This metric ranks
    the components based on 5 metrics and then outputs the mean rank across the 5 metrics.
    Thus "meanmetricrank" is a slightly better description but d_table was used in earlier
    versions of this code. It might be worth consistently using the same term, but this
    note will suffice for now.

    """

    function_name_idx = f"Step {selector.current_node_idx}: calc_max_good_meanmetricrank"

    if (
        (metric_suffix is not None)
        and (metric_suffix is not "")
        and isinstance(metric_suffix, str)
    ):
        metric_name = f"max_good_meanmetricrank_{metric_suffix}"
    else:
        metric_name = "max_good_meanmetricrank"

    outputs = {
        "decision_node_idx": selector.current_node_idx,
        "node_label": None,
        metric_name: None,
        "used_metrics": set(),
        "calc_cross_comp_metrics": [metric_name],
    }

    if only_used_metrics:
        return outputs["used_metrics"]

    if metric_name in selector.cross_component_metrics:
        LGR.warning(
            f"max_good_meanmetricrank already calculated. Overwriting previous value in {function_name_idx}"
        )

    if custom_node_label:
        outputs["node_label"] = custom_node_label
    else:
        outputs["node_label"] = f"Calc {metric_name}"

    if log_extra_info:
        LGR.info(log_extra_info)
    if log_extra_report:
        RepLGR.info(log_extra_report)

    comps2use = selectcomps2use(selector, decide_comps)
    confirm_metrics_exist(
        selector.component_table, outputs["used_metrics"], function_name=function_name_idx
    )

    if not comps2use:
        log_decision_tree_step(
            function_name_idx,
            comps2use,
            decide_comps=decide_comps,
        )
    else:

        num_prov_accept = len(comps2use)
        if "extend_factor" in selector.cross_component_metrics:
            extend_factor = selector.cross_component_metrics["extend_factor"]
            outputs[metric_name] = extend_factor * num_prov_accept
        else:
            raise ValueError(
                f"extend_factor needs to be in cross_component_metrics for {function_name_idx}"
            )

        selector.cross_component_metrics[metric_name] = outputs[metric_name]

        log_decision_tree_step(function_name_idx, comps2use, calc_outputs=outputs)

    selector.tree["nodes"][selector.current_node_idx]["outputs"] = outputs

    return selector


calc_max_good_meanmetricrank.__doc__ = calc_max_good_meanmetricrank.__doc__.format(**decision_docs)


def calc_varex_kappa_ratio(
    selector,
    decide_comps,
    log_extra_report="",
    log_extra_info="",
    custom_node_label="",
    only_used_metrics=False,
):
    """
    Calculates the variance explained / kappa ratio for the componentse in decide_comps
    and add those values to a new column in the component_table titled "varex kappa ratio".
    Also calculated kappa_rate which is a cross_component_metric

    Parameters
    ----------
    {selector}
    {decide_comps}
    {log_extra}
    {custom_node_label}
    {only_used_metrics}

    Returns
    -------
    {basicreturns}

    Note
    ----
    These measures are used in the original kundu decision tree.
    kappa_rate = (max-min kappa values of selected components)/(max-min variance explained)
    varex_k
    varex kappa ratio = kappa_rate * "variance explained"/"kappa" for each component
    Components with larger variance and smaller kappa are more likely to be rejected
    This metric sometimes causes issues with high magnitude BOLD responses
    such as the V1 response to a block-design flashing checkerboard
    """

    function_name_idx = f"Step {selector.current_node_idx}: calc_varex_kappa_ratio"

    outputs = {
        "decision_node_idx": selector.current_node_idx,
        "node_label": None,
        "kappa_rate": None,
        "used_metrics": {"kappa", "variance explained"},
        "calc_cross_comp_metrics": ["kappa_rate"],
        "added_component_table_metrics": ["varex kappa ratio"],
    }

    if only_used_metrics:
        return outputs["used_metrics"]

    if "kappa_rate" in selector.cross_component_metrics:
        LGR.warning(
            f"kappa_rate already calculated. Overwriting previous value in {function_name_idx}"
        )

    if "varex kappa ratio" in selector.component_table:
        raise ValueError(
            f"'varex kappa ratio' is already a column in the component_table. Recalculating in {function_name_idx} can cause problems since these are only calculated on a subset of components"
        )

    if custom_node_label:
        outputs["node_label"] = custom_node_label
    else:
        outputs["node_label"] = "Calc varex kappa ratio"

    if log_extra_info:
        LGR.info(log_extra_info)
    if log_extra_report:
        RepLGR.info(log_extra_report)

    comps2use = selectcomps2use(selector, decide_comps)
    confirm_metrics_exist(
        selector.component_table, outputs["used_metrics"], function_name=function_name_idx
    )

    if not comps2use:
        log_decision_tree_step(
            function_name_idx,
            comps2use,
            decide_comps=decide_comps,
        )
    else:
        kappa_rate = (
            np.nanmax(selector.component_table.loc[comps2use, "kappa"])
            - np.nanmin(selector.component_table.loc[comps2use, "kappa"])
        ) / (
            np.nanmax(selector.component_table.loc[comps2use, "variance explained"])
            - np.nanmin(selector.component_table.loc[comps2use, "variance explained"])
        )
        outputs["kappa_rate"] = kappa_rate
        LGR.info(f"Kappa rate found to be {kappa_rate} from components " f"{comps2use}")
        selector.component_table["varex kappa ratio"] = (
            kappa_rate
            * selector.component_table.loc[comps2use, "variance explained"]
            / selector.component_table.loc[comps2use, "kappa"]
        )
        # Unclear if necessary, but this may clean up a weird issue on passing references in a data frame
        # See longer comment in selection_utils.comptable_classification_changer
        selector.component_table = selector.component_table.copy()

        selector.cross_component_metrics["kappa_rate"] = outputs["kappa_rate"]

        # Keep a list of added metrics to the component table in selector
        if hasattr(selector, "added_component_table_metrics"):
            selector.added_component_table_metrics.add(["varex kappa ratio"])
        else:
            selector.added_component_table_metrics = set(["varex kappa ratio"])

        log_decision_tree_step(function_name_idx, comps2use, calc_outputs=outputs)

    selector.tree["nodes"][selector.current_node_idx]["outputs"] = outputs

    return selector


calc_varex_kappa_ratio.__doc__ = calc_varex_kappa_ratio.__doc__.format(**decision_docs)


"""
EVERTYHING BELOW HERE IS FOR THE KUNDU DECISION TREE AND IS NOT YET UPDATED
"""


# def highvariance_highmeanmetricrank_highkapparatio(
#     comptable,
#     decision_node_idx,
#     ifTrue,
#     ifFalse,
#     decide_comps,
#     n_echos,
#     n_vols=None,
#     extend_factor=None,
#     restrict_factor=2,
#     prev_X_steps=0,
#     high_perc=90,
#     log_extra_report="",
#     log_extra_info="",
#     custom_node_label="",
#     only_used_metrics=False,
# ):
#     """
#     Finds components with variance above a threshold,
#     a mean metric rank above a threshold, and kappa ratio above a threshold.
#     This would typically be used to identify borderline remaining components to reject.

#     Parameters
#     ----------
#     {comptable}
#     {decision_node_idx}
#     {ifTrue}
#     {ifFalse}
#     {decide_comps}
#     {n_echos}
#     {n_vols}
#     {extend_factor}
#     {restrict_factor}
#     {prev_X_steps}
#     {log_extra}
#     {custom_node_label}
#     {only_used_metrics}

#     Returns
#     -------
#     {basicreturns}
#     """

#     used_metrics = [
#         "variance explained",
#         "kappa",
#         "rho",
#         "dice_FT2",
#         "signal-noise_t",
#         "countsigFT2",
#         "countnoise",
#     ]
#     if only_used_metrics:
#         return used_metrics

#     function_name_idx = "Step {}: highvariance_highmeanmetricrank_highkapparatio".format(
#         decision_node_idx
#     )
#     if custom_node_label:
#         node_label = custom_node_label
#     else:
#         node_label = "highvar highmeanmetricrank highkapparatio"

#     if log_extra_info:
#         LGR.info(log_extra_info)
#     if log_extra_report:
#         RepLGR.info(log_extra_report)
#     metrics_exist, missing_metrics = confirm_metrics_exist(
#         comptable, used_metrics, function_name=function_name_idx
#     )

#     comps2use = selectcomps2use(comptable, decide_comps)
#     if not comps2use:
#         log_decision_tree_step(function_name_idx, comps2use, decide_comps=decide_comps)
#         numTrue = 0
#         numFalse = 0
#         dnode_outputs = create_dnode_outputs(decision_node_idx, used_metrics, node_label, 0, 0)
#     else:
#         # This will either identify a previously calculated revised meanmetricrank and
#         # return it or it will calculate a revised meanmetricrank, return it,
#         # and add a new column to comptable that contains this new metric
#         meanmetricrank, comptable = get_new_meanmetricrank(comptable, comps2use, decision_node_idx)

#         # Identify components that were either provionsally accepted
#         # or don't have a final classificaiton in prev_X_steps previous nodes
#         previous_comps2use = prev_classified_comps(
#             comptable,
#             decision_node_idx,
#             ["provisionalaccept", "provisionalreject", "unclassified"],
#             prev_X_steps=prev_X_steps,
#         )
#         previous_provaccept_comps2use = prev_classified_comps(
#             comptable,
#             decision_node_idx,
#             ["provisionalaccept"],
#             prev_X_steps=prev_X_steps,
#         )

#         kappa_elbow = kappa_elbow_kundu(comptable, n_echos)

#         # This should be the same as the MEICA 2.7 code except that I'm using provisionalaccept
#         # instead of >kappa_eblow and <rho_elbow, which si how provisionally accepted components
#         # are initially classified
#         num_acc_guess = int(
#             np.mean(
#                 [
#                     len(previous_provaccept_comps2use),
#                     np.sum(comptable.loc[previous_comps2use, "kappa"] > kappa_elbow),
#                 ]
#             )
#         )

#         # a scaling factor that is either based on the number of volumes or can be
#         # directly assigned
#         extend_factor = get_extend_factor(n_vols=n_vols, extend_factor=extend_factor)

#         varex_upper_thresh = scoreatpercentile(
#             comptable.loc[previous_provaccept_comps2use, "variance explained"],
#             high_perc,
#         )

#         # get kappa ratio
#         acc_prov = prev_classified_comps(
#             comptable,
#             decision_node_idx,
#             ["provisionalaccept"],
#             prev_X_steps=prev_X_steps,
#         )
#         kappa_rate = (
#             np.nanmax(comptable.loc[acc_prov, "kappa"])
#             - np.nanmin(comptable.loc[acc_prov, "kappa"])
#         ) / (
#             np.nanmax(comptable.loc[acc_prov, "variance explained"])
#             - np.nanmin(comptable.loc[acc_prov, "variance explained"])
#         )
#         LGR.info(f"Kappa rate found to be {kappa_rate} from components " f"{acc_prov}")
#         comptable["kappa ratio"] = (
#             kappa_rate * comptable["variance explained"] / comptable["kappa"]
#         )

#         conservative_guess = num_acc_guess / restrict_factor
#         db_mmrank = meanmetricrank.loc[comps2use] > conservative_guess
#         db_kapparatio = comptable.loc[comps2use, "kappa ratio"] > (extend_factor * 2)
#         db_var_upper = comptable.loc[comps2use, "variance explained"] > (
#             varex_upper_thresh * extend_factor
#         )
#         decision_boolean = db_mmrank & db_kapparatio & db_var_upper

#         comptable = change_comptable_classifications(
#             comptable, ifTrue, ifFalse, decision_boolean, str(decision_node_idx)
#         )
#         numTrue = np.asarray(decision_boolean).sum()
#         numFalse = np.logical_not(decision_boolean).sum()
#         # print(('numTrue={}, numFalse={}, numcomps2use={}'.format(
#         #        numTrue, numFalse, len(comps2use))))

#         log_decision_tree_step(
#             function_name_idx,
#             comps2use,
#             numTrue=numTrue,
#             numFalse=numFalse,
#             ifTrue=ifTrue,
#             ifFalse=ifFalse,
#         )

#         dnode_outputs = create_dnode_outputs(
#             decision_node_idx,
#             used_metrics,
#             node_label,
#             numTrue,
#             numFalse,
#             n_echos=n_echos,
#             n_vols=n_vols,
#             varex_threshold=varex_upper_thresh,
#             restrict_factor=2,
#             prev_X_steps=prev_X_steps,
#             max_good_meanmetricrank=conservative_guess,
#             num_acc_guess=num_acc_guess,
#             extend_factor=extend_factor,
#         )

#     return comptable, dnode_outputs


# highvariance_highmeanmetricrank_highkapparatio.__doc__ = (
#     highvariance_highmeanmetricrank_highkapparatio.__doc__.format(**decision_docs)
# )


# def highvariance_highmeanmetricrank(
#     comptable,
#     decision_node_idx,
#     ifTrue,
#     ifFalse,
#     decide_comps,
#     n_echos,
#     n_vols=None,
#     low_perc=25,
#     high_perc=90,
#     extend_factor=None,
#     prev_X_steps=0,
#     recalc_varex_lower_thresh=False,
#     log_extra_report="",
#     log_extra_info="",
#     custom_node_label="",
#     only_used_metrics=False,
# ):
#     """
#     Finds components with variance above a threshold,
#     a mean metric rank above a threshold, and kappa ratio above a threshold.
#     This would typically be used to identify borderline remaining components to reject.

#     Parameters
#     ----------
#     {comptable}
#     {decision_node_idx}
#     {ifTrue}
#     {ifFalse}
#     {decide_comps}
#     {n_echos}
#     {n_vols}
#     {extend_factor}
#     {prev_X_steps}
#     {log_extra}
#     {custom_node_label}
#     {only_used_metrics}

#     Returns
#     -------
#     {basicreturns}
#     """

#     used_metrics = [
#         "variance explained",
#         "kappa",
#         "rho",
#         "dice_FT2",
#         "signal-noise_t",
#         "countsigFT2",
#         "countnoise",
#     ]
#     if only_used_metrics:
#         return used_metrics

#     function_name_idx = (
#         "Step {}: highvariance_highmeanmetricrank_highkapparatio".format(
#             (decision_node_idx)
#         )
#     )
#     if custom_node_label:
#         node_label = custom_node_label
#     else:
#         node_label = "highvar highmeanmetricrank highkapparatio"

#     if log_extra_info:
#         LGR.info(log_extra_info)
#     if log_extra_report:
#         RepLGR.info(log_extra_report)
#     metrics_exist, missing_metrics = confirm_metrics_exist(
#         comptable, used_metrics, function_name=function_name_idx
#     )

#     comps2use = selectcomps2use(comptable, decide_comps)
#     if not comps2use:
#         log_decision_tree_step(function_name_idx, comps2use, decide_comps=decide_comps)
#         numTrue = 0
#         numFalse = 0
#         dnode_outputs = create_dnode_outputs(
#             decision_node_idx, used_metrics, node_label, 0, 0
#         )
#     else:
#         # This will either identify a previously calculated revised meanmetricrank and
#         # return it or it will calculate a revised meanmetricrank, return it,
#         # and add a new column to comptable that contains this new metric
#         meanmetricrank, comptable = get_new_meanmetricrank(
#             comptable, comps2use, decision_node_idx
#         )

#         # Identify components that were either provionsally accepted
#         # or don't have a final classificaiton in prev_X_steps previous nodes
#         previous_comps2use = prev_classified_comps(
#             comptable,
#             decision_node_idx,
#             ["provisionalaccept", "provisionalreject", "unclassified"],
#             prev_X_steps=prev_X_steps,
#         )
#         previous_provaccept_comps2use = prev_classified_comps(
#             comptable,
#             decision_node_idx,
#             ["provisionalaccept"],
#             prev_X_steps=prev_X_steps,
#         )

#         kappa_elbow = kappa_elbow_kundu(comptable, n_echos)

#         # This should be the same as the MEICA 2.7 code except that I'm using provisionalaccept
#         # instead of >kappa_eblow and <rho_elbow, which si how provisionally accepted components
#         # are initially classified
#         num_acc_guess = int(
#             np.mean(
#                 len(previous_provaccept_comps2use),
#                 np.sum(comptable.loc[previous_comps2use, "kappa"] > kappa_elbow),
#             )
#         )

#         # a scaling factor that is either based on the number of volumes or can be
#         # directly assigned
#         extend_factor = get_extend_factor(n_vols=n_vols, extend_factor=extend_factor)

#         conservative_guess2 = num_acc_guess * high_perc / 100.0
#         db_mmrank = meanmetricrank.loc[comps2use] > conservative_guess2

#         if recalc_varex_lower_thresh:
#             # Note: In MEICA v2.7 code, the included components are:
#             # [comps2use[:num_acc_guess]]. That would only make sense if the
#             # components were sorted by variance and I don't think they were.
#             # even still, this would be the the same as shifting the percentile
#             # based on num_acc_guess. The threshold without num_acc_guess seems
#             # equally arbitry so just keeping that for simplicity.
#             varex_lower_thresh = scoreatpercentile(
#                 comptable.loc[comps2use, "variance explained"], low_perc
#             )
#         else:
#             varex_lower_thresh = scoreatpercentile(
#                 comptable.loc[previous_provaccept_comps2use, "variance explained"],
#                 low_perc,
#             )

#         db_var_lower = comptable.loc[comps2use, "variance explained"] > (
#             varex_lower_thresh * extend_factor
#         )

#         decision_boolean = db_mmrank & db_var_lower

#         comptable = change_comptable_classifications(
#             comptable, ifTrue, ifFalse, decision_boolean, str(decision_node_idx)
#         )
#         numTrue = np.asarray(decision_boolean).sum()
#         numFalse = np.logical_not(decision_boolean).sum()
#         # print(('numTrue={}, numFalse={}, numcomps2use={}'.format(
#         #        numTrue, numFalse, len(comps2use))))

#         log_decision_tree_step(
#             function_name_idx,
#             comps2use,
#             numTrue=numTrue,
#             numFalse=numFalse,
#             ifTrue=ifTrue,
#             ifFalse=ifFalse,
#         )

#         dnode_outputs = create_dnode_outputs(
#             decision_node_idx,
#             used_metrics,
#             node_label,
#             numTrue,
#             numFalse,
#             n_echos=n_echos,
#             n_vols=n_vols,
#             varex_threshold=varex_lower_thresh,
#             prev_X_steps=prev_X_steps,
#             max_good_meanmetricrank=conservative_guess2,
#             num_acc_guess=num_acc_guess,
#             extend_factor=extend_factor,
#         )

#     return comptable, dnode_outputs


# highvariance_highmeanmetricrank.__doc__ = (
#     highvariance_highmeanmetricrank.__doc__.format(**decision_docs)
# )


# def highvariance_lowkappa(
#     comptable,
#     decision_node_idx,
#     ifTrue,
#     ifFalse,
#     decide_comps,
#     n_echos,
#     low_perc=25,
#     log_extra_report="",
#     log_extra_info="",
#     custom_node_label="",
#     only_used_metrics=False,
# ):
#     """
#     Finds components with variance above a threshold,
#     a mean metric rank above a threshold, and kappa ratio above a threshold.
#     This would typically be used to identify borderline remaining components to reject.

#     Parameters
#     ----------
#     {comptable}
#     {decision_node_idx}
#     {ifTrue}
#     {ifFalse}
#     {decide_comps}
#     {n_echos}
#     {log_extra}
#     {custom_node_label}
#     {only_used_metrics}

#     Returns
#     -------
#     {basicreturns}
#     """

#     used_metrics = ["variance explained", "kappa"]
#     if only_used_metrics:
#         return used_metrics

#     function_name_idx = "Step {}: highvariance_lowkappa".format(decision_node_idx)
#     if custom_node_label:
#         node_label = custom_node_label
#     else:
#         node_label = "highvariance lowkappa"

#     if log_extra_info:
#         LGR.info(log_extra_info)
#     if log_extra_report:
#         RepLGR.info(log_extra_report)
#     metrics_exist, missing_metrics = confirm_metrics_exist(
#         comptable, used_metrics, function_name=function_name_idx
#     )

#     comps2use = selectcomps2use(comptable, decide_comps)
#     if not comps2use:
#         log_decision_tree_step(function_name_idx, comps2use, decide_comps=decide_comps)
#         numTrue = 0
#         numFalse = 0
#         dnode_outputs = create_dnode_outputs(
#             decision_node_idx, used_metrics, node_label, 0, 0
#         )
#     else:
#         kappa_elbow = kappa_elbow_kundu(comptable, n_echos)
#         db_kappa = comptable.loc[comps2use, "kappa"] <= kappa_elbow

#         # Note: In MEICA v2.7 code, the included components are:
#         # [comps2use[:num_acc_guess]]. That would only make sense if the
#         # components were sorted by variance and I don't think they were.
#         # even still, this would be the the same as shifting the percentile
#         # based on num_acc_guess. The threshold without num_acc_guess seems
#         # equally arbitry so just keeping that for simplicity.
#         varex_lower_thresh = scoreatpercentile(
#             comptable.loc[comps2use, "variance explained"], low_perc
#         )

#         db_var_lower = (
#             comptable.loc[comps2use, "variance explained"] > varex_lower_thresh
#         )

#         decision_boolean = db_kappa & db_var_lower

#         comptable = change_comptable_classifications(
#             comptable, ifTrue, ifFalse, decision_boolean, str(decision_node_idx)
#         )
#         numTrue = np.asarray(decision_boolean).sum()
#         numFalse = np.logical_not(decision_boolean).sum()
#         # print(('numTrue={}, numFalse={}, numcomps2use={}'.format(
#         #        numTrue, numFalse, len(comps2use))))

#         log_decision_tree_step(
#             function_name_idx,
#             comps2use,
#             numTrue=numTrue,
#             numFalse=numFalse,
#             ifTrue=ifTrue,
#             ifFalse=ifFalse,
#         )

#         dnode_outputs = create_dnode_outputs(
#             decision_node_idx,
#             used_metrics,
#             node_label,
#             numTrue,
#             numFalse,
#             n_echos=n_echos,
#             varex_threshold=varex_lower_thresh,
#         )

#     return comptable, dnode_outputs


# highvariance_lowkappa.__doc__ = highvariance_lowkappa.__doc__.format(**decision_docs)
