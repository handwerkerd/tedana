"""
Utility functions for tedana.selection
"""

import logging
import re
import numpy as np
from tedana.stats import getfbounds
from tedana.metrics.dependence import generate_decision_table_score

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")
RefLGR = logging.getLogger("REFERENCES")

##############################################################
# Functions that are used for interacting with component_table
##############################################################


def selectcomps2use(selector, decide_comps):
    """
    Give a list of component numbers that fit the classification types in
    decide_comps.

    Parameters
    ----------
    selector: :obj:`tedana.selection.ComponentSelector`
        Only uses the component_table in this object
    decide_comps: :obj:`str` or :obj:`list[str]` or :obj:`list[int]`
        This is string or a list of strings describing what classifications
        of components to operate on, using default or intermediate_classification
        labels. For example: decide_comps='unclassified' means to operate only on
        unclassified components. The label 'all' will operate on all components
        regardess of classification. This can also be used to pass through a list
        of component indices to comps2use

    Returns
    -------
    comps2use: :obj:`list[int]`
        A list of component indices that should be used by a function
    component_table : (C x M) :obj:`pandas.DataFrame`
        Reference to component metric table in `selector`. One row for each
        component, with a column for each metric. Since the component table
        is assigned rather than copied, changes to this variable will change
        `selector.component_table`

     Note
    ----
    TODO: Number indexing should work here, but validator would not currently allow
    numbers to be assigned to a node in the ComponentSelector object. May want to make
    sure this option is acceissble through the class.
    TODO: If a list of component number throw an if a number isn't valid rather than crashing
    """

    component_table = selector.component_table
    if type(decide_comps) == str:
        decide_comps = [decide_comps]
    if decide_comps[0] == "all":
        # All components with any string in the classification field
        # are set to True
        comps2use = list(range(component_table.shape[0]))

    elif (type(decide_comps) == list) and (type(decide_comps[0]) == str):
        comps2use = []
        for didx in range(len(decide_comps)):
            newcomps2use = component_table.index[
                component_table["classification"] == decide_comps[didx]
            ].tolist()
            comps2use = list(set(comps2use + newcomps2use))
    else:
        # decide_comps is already a string of indices
        comps2use = decide_comps

    # If no components are selected, then return None.
    # The function that called this can check for None and exit before
    # attempting any computations on no data
    if not comps2use:
        comps2use = None

    return comps2use, component_table


def change_comptable_classifications(
    selector,
    ifTrue,
    ifFalse,
    decision_boolean,
    tag_ifTrue=None,
    tag_ifFalse=None,
    dont_warn_reclassify=False,
):
    """
    Given information on whether a decision critereon is true or false for each component
    change or don't change the component classification

    Parameters
    ----------
    selector: :obj:`tedana.selection.ComponentSelector`
        The attributes used are component_table, component_status_table, and
        current_node_idx
    ifTrue, ifFalse: :obj:`str`
        If the condition in this step is true or false, give the component
        the label in this string. Options are 'accepted', 'rejected',
        'nochange', or intermediate_classification labels predefined in the
        decision tree. If 'nochange' then don't change the current component
        classification
    decision_boolean: :obj:`pd.Series(bool)`
        A dataframe column of equal length to component_table where each value
        is True or False.
    tag_ifTrue, tag_ifFalse: :obj:`str`
        A string containing a label in classification_tags that will be added to
        the classification_tags column in component_table if a component is
        classified as true or false. default=None
    dont_warn_reclassify: :obj:`bool`
        If this function changes a component classification from accepted or
        rejected to something else, it gives a warning. If this is True, that
        warning is suppressed. default=False

    Returns
    -------
    selector: :obj:`tedana.selection.ComponentSelector`
        component_table["classifications"] will reflect any new
        classifications.
        component_status_table will have a new column titled
        "Node current_node_idx" that is a copy of the updated classifications
        column.
        component_table["classification_tags"] will be updated to include any
        new tags. Each tag should appear only once in the string and tags will
        be separated by commas.
    numTrue, numFalse: :obj:`int`
        The number of True and False components in decision_boolean

    Note
    ----
    If a classification is changed away from accepted or rejected and
    dont_warn_reclassify is False, then a warning is logged
    """

    selector = comptable_classification_changer(
        selector, True, ifTrue, decision_boolean, tag_ifTrue
    )
    selector = comptable_classification_changer(
        selector, False, ifFalse, decision_boolean, tag_ifFalse
    )

    selector.component_status_table[
        f"Node {selector.current_node_idx}"
    ] = selector.component_table["classification"]

    numTrue = decision_boolean.sum()
    numFalse = np.logical_not(decision_boolean).sum()
    return selector, numTrue, numFalse


def comptable_classification_changer(
    selector,
    boolstate,
    classify_if,
    decision_boolean,
    tag_if=None,
    dont_warn_reclassify=False,
):
    """
    Implement the component classification changes specified in
    change_comptable_classifications.

    Parameters
    ----------
    selector: :obj:`tedana.selection.ComponentSelector`
        The attributes used are component_table, component_status_table, and
        current_node_idx
    boolstate : :obj:`bool`
        Change classifications only for True or False components in
        decision_boolean based on this variable
    classify_if: :obj:`str`
        This should be if_True or if_False to match boolstate.
        If the condition in this step is true or false, give the component
        the label in this string. Options are 'accepted', 'rejected',
        'nochange', or intermediate_classification labels predefined in the
        decision tree. If 'nochange' then don't change the current component
        classification
    decision_boolean: :obj:`pd.Series(bool)`
        A dataframe column of equal length to component_table where each value
        is True or False.
    tag_if: :obj:`str`
        This should be tag_ifTrue or tag_ifFalse to match boolstate
        A string containing a label in classification_tags that will be added to
        the classification_tags column in component_table if a component is
        classified as true or false. default=None
    dont_warn_reclassify: :obj:`bool`
        If this function changes a component classification from accepted or
        rejected to something else, it gives a warning. If this is True, that
        warning is suppressed. default=False
    Returns
    -------
    selector: :obj:`tedana.selection.ComponentSelector`
        Operates on the True OR False componets depending on boolstate
        component_table["classifications"] will reflect any new
        classifications.
        component_status_table will have a new column titled
        "Node current_node_idx" that is a copy of the updated classifications
        column.
        component_table["classification_tags"] will be updated to include any
        new tags. Each tag should appear only once in the string and tags will
        be separated by commas.
    If a classification is changed away from accepted or rejected and
    dont_warn_reclassify is False, then a warning is logged
    """

    if classify_if != "nochange":
        changeidx = decision_boolean.index[np.asarray(decision_boolean) == boolstate]
        current_classifications = set(
            selector.component_table.loc[changeidx, "classification"].tolist()
        )
        if current_classifications.intersection({"accepted", "rejected"}):
            if not dont_warn_reclassify:
                # don't make a warning if classify_if matches the current classification
                # That is reject->reject shouldn't throw a warning
                if (("accepted" in current_classifications) and (classify_if != "accepted")) or (
                    ("rejected" in current_classifications) and (classify_if != "rejected")
                ):
                    LGR.warning(
                        f"Step {selector.current_node_idx}: Some classifications are"
                        " changing away from accepted or rejected. Once a component is "
                        "accepted or rejected, it shouldn't be reclassified"
                    )
        selector.component_table.loc[changeidx, "classification"] = classify_if

        for idx in changeidx:
            tmpstr = selector.component_table.loc[idx, "classification_tags"]
            if tmpstr != "":
                tmpset = set(tmpstr.split(","))
                tmpset.update([tag_if])
            else:
                tmpset = set([tag_if])
            selector.component_table.loc[idx, "classification_tags"] = ",".join(
                str(s) for s in tmpset
            )
    return selector


def clean_dataframe(component_table):
    """
    Reorder columns in component table so that "classification"
    and "classification_tags" are last.

    Parameters
    ----------
    component_table : (C x M) :obj:`pandas.DataFrame`
        Component metric table. One row for each component, with a column for
        each metric

    Returns
    -------
    component_table : (C x M) :obj:`pandas.DataFrame`
        Same data as input, but the final two columns are "classification"
        and "classification_tags"
    """
    cols_at_end = ["classification", "classification_tags"]
    component_table = component_table[
        [c for c in component_table if c not in cols_at_end]
        + [c for c in cols_at_end if c in component_table]
    ]

    return component_table


LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")
RefLGR = logging.getLogger("REFERENCES")

#################################################
# Functions to validate inputs or log information
#################################################


def confirm_metrics_exist(component_table, necessary_metrics, function_name=None):
    """
    Confirm that all metrics declared in necessary_metrics are
    already included in comptable.

    Parameters
    ----------
    component_table : (C x M) :obj:`pandas.DataFrame`
            Component metric table. One row for each component, with a column for
            each metric. The index should be the component number.
    necessary_metrics : :obj:`set` a set of strings of metrics
    function_name : :obj:`str`
        Text identifying the function name that called this function

    Returns
    -------
    metrics_exist : :obj:`bool`
            True if all metrics in necessary_metrics are in component_table

    If metrics_exist is False then raise an error and end the program

    Notes
    -----
    This doesn't check if there are data in each metric's column, just that
    the columns exist. Also, this requires identical strings for the names
    of the metrics in necessary_metrics and the column labels in component_table
    """

    missing_metrics = necessary_metrics - set(component_table.columns)
    metrics_exist = len(missing_metrics) > 0
    if metrics_exist is True:
        if function_name is None:
            function_name = "unknown function"

        error_msg = (
            f"Necessary metrics for {function_name}: "
            f"{necessary_metrics}. "
            f"Comptable metrics: {set(component_table.columns)}. "
            f"MISSING METRICS: {missing_metrics}."
        )
        raise ValueError(error_msg)

    return metrics_exist


def log_decision_tree_step(
    function_name_idx,
    comps2use,
    decide_comps=None,
    numTrue=None,
    numFalse=None,
    ifTrue=None,
    ifFalse=None,
    calc_outputs=None,
):
    """
    Logging text to add for every decision tree calculation

    Parameters
    ----------
    function_name_idx: :obj:`str`
        The name of the function that should be logged. By convention, this
        be "Step current_node_idx: function_name"
    comps2use: :obj:`list[int]`
        A list of component indices that should be used by a function.
        Only used to report no components found if empty and report
        the number of components found if not empty.
    decide_comps: :obj:`str` or :obj:`list[str]` or :obj:`list[int]`
        This is string or a list of strings describing what classifications
        of components to operate on. Only used in this function to report
        its contents if no components with these classifications were found
    numTrue, numFalse: :obj:`int`
        The number of components classified as True or False
    ifTrue, ifFalse: :obj:`str`
        If a component is true or false, the classification to assign that
        component
    calc_outputs: :obj:`bool`
        True if the function being logged calculated new cross component
        metrics. If true, then log the metrics calculated and their values

    Returns
    -------
    Information is added to the LGR.info logger. This either logs that
    nothing was changed, the number of components classified as true or
    false and what they changed to, or the cross component metrics that were
    calculated
    """

    if comps2use is None:
        LGR.info(
            f"{function_name_idx} not applied because no remaining components were "
            f"classified as {decide_comps}"
        )
    if ifTrue or ifFalse:
        LGR.info(
            f"{function_name_idx} applied to {len(comps2use)} components. "
            f"{numTrue} True -> {ifTrue}. "
            f"{numFalse} False -> {ifFalse}."
        )
    if calc_outputs:
        calc_summaries = [
            f"{metric_name}={calc_outputs[metric_name]}"
            for metric_name in calc_outputs["calc_cross_comp_metrics"]
        ]
        LGR.info(f"{function_name_idx} calculated: {', '.join(calc_summaries)}")


def log_classification_counts(decision_node_idx, component_table):
    """
    Log the total counts for each component classification in component_table

    Parameters
    ----------
    decision_node_idx : :obj:`int`
        The index number for the function in the decision tree that just
        finished executing
    component_table : (C x M) :obj:`pandas.DataFrame`
        Component metric table. One row for each component, with a column for
        each metric. Only the "classification" column is usd in this function

    Returns
    -------
    The info logger will add a line like:
    'Step 4: Total component classifications: 10 accepted, 5 provisionalreject, 8 rejected'
    """

    (classification_labels, label_counts) = np.unique(
        component_table["classification"].values, return_counts=True
    )
    label_summaries = [
        f"{label_counts[i]} {label}" for i, label in enumerate(classification_labels)
    ]
    prelude = f"Step {decision_node_idx}: Total component classifications:"
    out_str = f"{prelude} {', '.join(label_summaries)}"
    LGR.info(out_str)


#######################################################
# Calculations that are used in decision tree functions
#######################################################
def getelbow_cons(arr, return_val=False):
    """
    Elbow using mean/variance method - conservative

    Parameters
    ----------
    arr : (C,) array_like
        Metric (e.g., Kappa or Rho) values.
    return_val : :obj:`bool`, optional
        Return the value of the elbow instead of the index. Default: False

    Returns
    -------
    :obj:`int` or :obj:`float`
        Either the elbow index (if return_val is True) or the values at the
        elbow index (if return_val is False)
    """
    if arr.ndim != 1:
        raise ValueError("Parameter arr should be 1d, not {0}d".format(arr.ndim))

    if not arr.size:
        raise ValueError(
            "Empty array detected during elbow calculation. "
            "This error happens when getelbow_cons is incorrectly called on no components. "
            "If you see this message, please open an issue at "
            "https://github.com/ME-ICA/tedana/issues with the full traceback and any data "
            "necessary to reproduce this error, so that we create additional data checks to "
            "prevent this from happening."
        )

    arr = np.sort(arr)[::-1]
    nk = len(arr)
    temp1 = [
        (arr[nk - 5 - ii - 1] > arr[nk - 5 - ii : nk].mean() + 2 * arr[nk - 5 - ii : nk].std())
        for ii in range(nk - 5)
    ]
    ds = np.array(temp1[::-1], dtype=np.int)
    dsum = []
    c_ = 0
    for d_ in ds:
        c_ = (c_ + d_) * d_
        dsum.append(c_)
    e2 = np.argmax(np.array(dsum))
    elind = np.max([getelbow(arr), e2])

    if return_val:
        return arr[elind]
    else:
        return elind


def getelbow(arr, return_val=False):
    """
    Elbow using linear projection method - moderate

    Parameters
    ----------
    arr : (C,) array_like
        Metric (e.g., Kappa or Rho) values.
    return_val : :obj:`bool`, optional
        Return the value of the elbow instead of the index. Default: False

    Returns
    -------
    :obj:`int` or :obj:`float`
        Either the elbow index (if return_val is True) or the values at the
        elbow index (if return_val is False)
    """
    if arr.ndim != 1:
        raise ValueError("Parameter arr should be 1d, not {0}d".format(arr.ndim))

    if not arr.size:
        raise ValueError(
            "Empty array detected during elbow calculation. "
            "This error happens when getelbow is incorrectly called on no components. "
            "If you see this message, please open an issue at "
            "https://github.com/ME-ICA/tedana/issues with the full traceback and any data "
            "necessary to reproduce this error, so that we create additional data checks to "
            "prevent this from happening."
        )

    arr = np.sort(arr)[::-1]
    n_components = arr.shape[0]
    coords = np.array([np.arange(n_components), arr])
    p = coords - coords[:, 0].reshape(2, 1)
    b = p[:, -1]
    b_hat = np.reshape(b / np.sqrt((b**2).sum()), (2, 1))
    proj_p_b = p - np.dot(b_hat.T, p) * np.tile(b_hat, (1, n_components))
    d = np.sqrt((proj_p_b**2).sum(axis=0))
    k_min_ind = d.argmax()

    if return_val:
        return arr[k_min_ind]
    else:
        return k_min_ind


def kappa_elbow_kundu(comptable, n_echos):
    """
    Calculate an elbow for kappa using the approach originally in
    Prantik Kundu's MEICA v2.7 code

    Parameters
    ----------
    comptable : (C x M) :obj:`pandas.DataFrame`
        Component metric table. One row for each component, with a column for
        each metric. The index should be the component number.
        Only the 'kappa' column is used in this function
    n_echos: :obj:`int`
        The number of echos in the multi-echo data

    Returns
    -------
    kappa_elbow: :obj:`float`
        The 'elbow' value for kappa values, above which components are considered
        more likely to contain T2* weighted signals
    """
    # low kappa threshold
    f05, _, f01 = getfbounds(n_echos)
    # get kappa values for components below a significance threshold
    kappas_nonsig = comptable.loc[comptable["kappa"] < f01, "kappa"]

    # Would an elbow from all Kappa values *ever* be lower than one from
    # a subset of lower values?
    # Note: Only use the subset of values if it includes at least 5 data point
    #  That is enough to calculate an elbow of a curve
    #  This is an arbitrary threshold not from the original meica as is
    #  worth reconsidering at some point
    if kappas_nonsig.size > 5:
        kappa_elbow = np.min(
            (
                getelbow(kappas_nonsig, return_val=True),
                getelbow(comptable["kappa"], return_val=True),
            )
        )
        LGR.info(("Calculating kappa elbow based on min of all and nonsig components."))
    else:
        kappa_elbow = getelbow(comptable["kappa"], return_val=True)
        LGR.info(("Calculating kappa elbow based on all components."))

    return kappa_elbow


def get_extend_factor(n_vols=None, extend_factor=None):
    """
    extend_factor is a scaler used to set a threshold for the d_table_score
    It is either defined by the number of volumes in the time series or if directly
    defined by the user. If it is defined by the user, that takes precedence over
    using the number of volumes in a calculation

    Parameters
    ----------
    n_vols: :obj:`int`
        The number of volumes in an fMRI time series. default=None
        In the MEICA code, extend_factor was hard-coded to 2 for data with more
        than 100 volumes and 3 for data with less than 100 volumes.
        Now is linearly ramped from 2-3 for vols between 90 & 110

    extend_factor: :obj:`float`
        The scaler used to set a threshold for d_table_score. default=None

    Returns
    -------
    extend_factor: :obj:`float`

    Note
    ----
    Either n_vols OR extend_factor is a required input
    """

    if extend_factor:
        LGR.info("extend_factor={}, as defined by user".format(extend_factor))
    elif n_vols:
        if n_vols < 90:
            extend_factor = 3
        elif n_vols < 110:
            extend_factor = 2 + (n_vols - 90) / 20
        else:
            extend_factor = 2
        LGR.info("extend_factor={}, based on number of fMRI volumes".format(extend_factor))
    else:
        error_msg = "get_extend_factor need n_vols or extend_factor as an input"
        LGR.error(error_msg)
        ValueError(error_msg)
    return extend_factor


def get_new_meanmetricrank(comptable, comps2use, decision_node_idx, calc_new_rank=False):
    """
    If a revised d_table_score was already calculated, use that.
    If not, calculate a new d_table_score based on the components
    identified in comps2use

    Parameters
    ----------
    comptable
    comps2use
    decision_node_idx: :obj:`int`
        The index for the current decision node
    calc_new_rank: :obj:`bool`
        calculate a new d_table_score even if a revised mean
        metric rank was already calculated

    Return
    ------
    meanmetricrank
    comptable
    """
    rank_label = "d_table_score" + str(decision_node_idx)
    if not calc_new_rank and (rank_label in comptable.columns):
        # go ahead and return existing
        return comptable[rank_label], comptable
    # get the array of ranks
    ranks = generate_decision_table_score(
        comptable.loc[comps2use, "kappa"],
        comptable.loc[comps2use, "dice_FT2"],
        comptable.loc[comps2use, "signal-noise_t"],
        comptable.loc[comps2use, "countnoise"],
        comptable.loc[comps2use, "countsigFT2"],
    )
    # see if we need to make a new column
    if rank_label not in comptable.columns:
        comptable[rank_label] = np.zeros(comptable.shape[0]) * np.nan

    # fill in the column with the components of interest
    for c, rank in zip(comps2use, ranks):
        comptable[c, rank_label] = rank

    return comptable[rank_label], comptable


def prev_classified_comps(comptable, decision_node_idx, classification_label, prev_X_steps=0):
    """
    Output a list of components with a specific label during the current or
    previous X steps of the decision tree. For example, if
    classification_label = ['provisionalaccept'] and prev_X_steps = 0
    then this outputs the indices of components that are currenlty
    classsified as provisionalaccept. If prev_X_steps=2, then this will
    output components that are classified as provisionalaccept or were
    classified as such any time before the previous two decision tree steps

    Parameters
    ----------
    comptable
    n_echos: :obj:`int`
        The number of echos in the multi-echo data set
    decision_node_idx: :obj:`int`
        The index of the node in the decision tree that called this function
    classification_label: :obj:`list[str]`
        A list of strings containing classification labels to identify in components
        For example: ['provisionalaccept']
    prev_X_steps: :obj:`int`
        If 0, then just count the number of provisionally accepted or rejected
        or unclassified components in the current node. If this is a positive
        integer, then also check if a component was a in one of those three
        categories in ignore_prev_X_steps previous nodes. default=0

    Returns
    -------
    full_comps2use: :obj:`list[int]`
        A list of indices of components that have or add classification_lable
    """

    full_comps2use = selectcomps2use(comptable, classification_label)
    rationales = comptable["rationale"]

    if prev_X_steps > 0:  # if checking classifications in prevision nodes
        for compidx in range(len(comptable)):
            tmp_rationale = rationales.values[compidx]
            tmp_list = re.split(":|;| ", tmp_rationale)
            while "" in tmp_list:  # remove blank strings after splitting rationale
                tmp_list.remove("")
            # Check the previous nodes
            # This is inefficient, but it should work
            for didx in range(max(0, decision_node_idx - prev_X_steps), decision_node_idx):
                if str(didx) in tmp_list:
                    didx_loc = tmp_list.index(str(didx))
                    if didx_loc > 1:
                        tmp_classifier = tmp_list[didx_loc - 1]
                        if tmp_classifier in classification_label:
                            full_comps2use.append(compidx)

    full_comps2use = list(set(full_comps2use))

    return full_comps2use