"""Toy implementation of DecisionBoard, Node classes"""

import pandas as pd

UNCLASSIFIED = "Unclassified"


def toy_comptable():
    """Returns a toy comptable

    Returns
    -------
    pd.DataFrame with Component, kappa, and rho for 5 components
    """
    return pd.DataFrame(
        {
            "Component": [i for i in range(5)],
            "kappa": [10, 9, 8, 3, 1],
            "rho": [1, 3, 8, 9, 10],
        }
    )


class DecisionBoard:
    """A class for tracking decisions made about components.

    Attributes
    ----------
    _component_table: pd.DataFrame
        A table with one metric per component.
    _global_metrics: dict
        A dictionary where one value applies to all components.
    _status_table: pd.DataFrame
        A table with each component's status provenance.
    _n_steps: int
        The number of steps run so far
    _nodes: list(Node)
        The list of decision nodes to be run.

    Methods
    -------
    __init__
    select
    set_status
    set_global
    set_metrics
    get_required_metrics
    run

    Notes
    -----
    The DecisionBoard API enforces the following rules:
        1. No values may be overwritten.
        2. Selection is performed only on current component statuses.
    When the DecisionBoard is initialized, it constructs a list of Node
    objects which may only make simple calls to this API. It uses the
    initialization routine to determine if the supplied component table
    and node order will be valid at runtime, so it may crash early.
    It is the responsibility of a Node method author to ensure API
    compatibility.

    See Also
    --------
    Node
    """
    def __init__(self, component_table, specification):
        """Construct a DecisionBoard instance

        Parameters
        ----------
        comptable: pd.DataFrame
            A dataframe containing components and metrics
        specification: dict
            A specification for building nodes; see Node docoumentation

        Raises
        ------
        TypeError, if types are not adhered to
        RuntimeError, if the specification is illegal
        """
        self._component_table = component_table
        self._global_metrics = {}
        self._status_table = pd.DataFrame(self._component_table["Component"])
        self._n_steps = 0
        self._status_table.insert(1, f"Step {self._n_steps}", UNCLASSIFIED)
    def select(self, metrics, status):
        """Select and return a sub-table matching the metrics and status

        Parameters
        ----------
        metrics: list(str)
            The metrics to include in the returned sub-table
        status: list(str)
            The statuses to select

        Returns
        -------
        pd.DataFrame with all matching components and metrics

        Notes
        -----
        Requires metric specification to make sure that Nodes are
        compliant with the comptable; if the entire component table were
        returned, Nodes would not be forced to only use the metrics they
        ask for and it would be possible to de facto require un-specified
        metrics. This forces a RuntimeError instead, to aid in the
        debugging of decision trees with incorrectly specified
        dependencies.
        """
        if not isinstance(metrics, list):
            raise TypeError("Metrics must be supplied as a list")
        if not isinstance(status, list):
            raise TypeError("Statuses must be supplied as a list")
        if "Component" not in metrics:
            metrics.insert(0, "Component")
        tail_label = f"Step {self._n_steps}"
        status_matches = self._status_table[tail_label].isin(status)
        # NOTE: status and component table must have same number of rows
        return self._component_table[status_matches][metrics]

