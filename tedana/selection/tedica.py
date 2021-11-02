"""
Functions to identify TE-dependent and TE-independent components.
"""
import logging
import numpy as np
from scipy import stats

from tedana.stats import getfbounds
from tedana.selection.DecisionTree import DecisionTree
from tedana.selection._utils import clean_dataframe
from tedana.metrics import collect

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")
RefLGR = logging.getLogger("REFERENCES")


def manual_selection(comptable, acc=None, rej=None):
    """
    Perform manual selection of components.

    Parameters
    ----------
    comptable : (C x M) :obj:`pandas.DataFrame`
        Component metric table, where `C` is components and `M` is metrics
    acc : :obj:`list`, optional
        List of accepted components. Default is None.
    rej : :obj:`list`, optional
        List of rejected components. Default is None.

    Returns
    -------
    comptable : (C x M) :obj:`pandas.DataFrame`
        Component metric table with classification.
    metric_metadata : :obj:`dict`
        Dictionary with metadata about calculated metrics.
        Each entry corresponds to a column in ``comptable``.
    """
    LGR.info("Performing manual ICA component selection")
    RepLGR.info(
        "Next, components were manually classified as "
        "BOLD (TE-dependent), non-BOLD (TE-independent), or "
        "uncertain (low-variance)."
    )
    if (
        "classification" in comptable.columns
        and "original_classification" not in comptable.columns
    ):
        comptable["original_classification"] = comptable["classification"]
        comptable["original_rationale"] = comptable["rationale"]

    comptable["classification"] = "accepted"
    comptable["rationale"] = ""

    all_comps = comptable.index.values
    if acc is not None:
        acc = [int(comp) for comp in acc]

    if rej is not None:
        rej = [int(comp) for comp in rej]

    if acc is not None and rej is None:
        rej = sorted(np.setdiff1d(all_comps, acc))
    elif acc is None and rej is not None:
        acc = sorted(np.setdiff1d(all_comps, rej))
    elif acc is None and rej is None:
        LGR.info(
            "No manually accepted or rejected components supplied. "
            "Accepting all components."
        )
        # Accept all components if no manual selection provided
        acc = all_comps[:]
        rej = []

    ign = np.setdiff1d(all_comps, np.union1d(acc, rej))
    comptable.loc[acc, "classification"] = "accepted"
    comptable.loc[rej, "classification"] = "rejected"
    comptable.loc[rej, "rationale"] += "I001;"
    comptable.loc[ign, "classification"] = "ignored"
    comptable.loc[ign, "rationale"] += "I001;"

    # Move decision columns to end
    comptable = clean_dataframe(comptable)
    metric_metadata = collect.get_metadata(comptable)
    return comptable, metric_metadata


def automatic_selection(comptable, n_echos, n_vols, tree="simple"):
    """Classify components based on component table and tree type.

    Parameters
    ----------
    comptable: pd.DataFrame
        The component table to classify
    n_echos: int
        The number of echoes in this dataset
    tree: str
        The type of tree to use for the DecisionTree object

    Returns
    -------
    A dataframe of the component table, after classification and reorder
    The metadata associated with the component table

    See Also
    --------
    DecisionTree, the class used to represent the classification process
    """
    comptable["rationale"] = ""
    dt = DecisionTree(tree, comptable, n_echos=n_echos, n_vols=n_vols)
    dt.run()
    dt.metadata = collect.get_metadata(dt.component_table)

    # TODO: Eventually return just dt
    return (
        dt.component_table,
        dt.cross_component_metrics,
        dt.component_status_table,
        dt.metadata,
    )
