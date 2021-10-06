"""Toy implementation of DecisionBoard, Node classes"""

import pandas as pd

UNCLASSIFIED = "Unclassified"

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
        self._status_table.insert(1, "Step 0", UNCLASSIFIED)
