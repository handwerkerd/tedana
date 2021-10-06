"""Toy implementation of DecisionBoard, Node classes"""

from collections import OrderedDict

import pandas as pd

UNCLASSIFIED = "Unclassified"
REJECTED = "Rejected"
ACCEPTED = "Accepted"


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

def toy_nodeset():
    """Returns a toy nodeset

    Returns
    -------
    OrderedDict of nodes to be run
    """
    return OrderedDict([
        (
            "Reject low kappa",
            {
                "function": "metric_left_op_right",
                "left": "kappa",
                "op": "<",
                "right": "rho",
                "select": UNCLASSIFIED,
                "set_true_status": REJECTED,
            }
        ), (
            "Accept very high kappa",
            {
                "function": "metric_left_op_right",
                "left": "kappa",
                "op": ">",
                "right": "rho",
                "scale_right": 3,
                "select": UNCLASSIFIED,
                "set_true_status": ACCEPTED,
            },
        )
    ])


# TODO: implement label recording for each step
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
    _labeler: function(int)
        A pointer to a function to generate a label for a given step n
    _nodes: list(Node)
        The list of decision nodes to be run.
    _node_labels: list(str)
        The list of node labels

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
        # TODO: set so that component table is initially None, and added
        # later so that we can get what metrics are required up-front.
        self._component_table = component_table
        self._global_metrics = {}
        self._status_table = pd.DataFrame(self._component_table["Component"])
        self._n_steps = 0
        self._labeler = lambda x: f"Step {x}"
        self._status_table.insert(1, f"Step {self._n_steps}", UNCLASSIFIED)
        if len(specification) == 0:
            raise ValueError("A specification is required to have steps")
        self._nodes = []
        self._node_labels = []
        possible_statuses = set([UNCLASSIFIED])
        available_metrics = set(self._component_table.columns)
        self._required_metrics = set()
        for label in specification:
            self._node_labels.append(label)
            function_spec = specification[label]
            nd = DecisionNode(self, function_spec)
            # TODO: add information about step label, number in errors
            for status in nd.selects_statuses():
                if status not in possible_statuses:
                    # TODO: make warning
                    print(
                        f"Status {status} is selected but does not yet "
                        "exist"
                    )
            for metric in nd.required_metrics():
                if metric not in available_metrics:
                    raise ValueError(
                        f"Metric {metric} is not available"
                    )
                self._required_metrics.add(metric)
            self._nodes.append(nd)
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
        tail_label = self._labeler(self._n_steps)
        status_matches = self._status_table[tail_label].isin(status)
        # NOTE: status and component table must have same number of rows
        return self._component_table[status_matches][metrics]
    def set_status(self, components, status):
        """Select a set of components and update their status

        Parameters
        ----------
        components: list(int)
            The list of integer component IDs to update status for
        status: str
            The status to update the components to
        """
        if not isinstance(components, list):
            raise TypeError("Components to set status must be a list")
        if not isinstance(status, str):
            raise TypeError("Status must be a string")
        self._n_steps += 1
        curr_label = self._labeler(self._n_steps)
        prev_label = self._labeler(self._n_steps - 1)
        self._status_table.insert(self._n_steps + 1, curr_label, "")
        self._status_table[curr_label] = self._status_table[prev_label]
        select_comps = self._status_table["Component"].isin(components)
        self._status_table.loc[select_comps, curr_label] = status
    def set_global(self, name, value):
        """Set a component-wide global value

        Parameters
        ----------
        name: str
            The name of the global to set
        value: float, int
            The value of the global to set

        Raises
        ------
        RuntimeError, if the global's name is already occupied
        """
        if not isinstance(name, str):
            raise TypeError("Global values must be of type str")
        if not isinstance(value, float) and not isinstance(value, int):
            raise TypeError(
                "Global values must be scalar numbers; got: "
                f"{type(value)}"
            )
        if name in self._global_metrics:
            # NOTE: when building from nodes this is already checked
            # This is here to uphold DecisionBoard's contract in case
            # manual manipulations of the board are performed.
            raise RuntimeError(
                f"The name supplied {name} is already in use"
            )
        self._global_metrics[name] = value
    def set_metrics(self, ids, values):
        """Sets the entire component

        Parameters
        ----------
        ids: list(int)
            The components to set value of; unlisted will receive NaN
        values: list(number)
            The values to assign pairwise with components
        """
        # TODO: discuss implementation, API for sensibility
        # TODO: implement
        raise RuntimeError("This function is unimplemented; sorry")
    def run(self):
        for node in self._nodes:
            # TODO: log information
            node.run()


class DecisionNode:
    """A small Actor designed to interface with a DecisionBoard for
    automatic decisions

    Attributes
    ----------
    _board: DecisionBoard
        The board that this node will interface with
    _fn: function pointer
        A function that will be run with this Node
    _required_metrics: list(str)
        The list of metrics this node requires
    _required_global: None or str
        The global this node requires
    _produces_global: None or str
        The global that this node produces
    _sets_status: list(str)
        The status that this node may set for some components
    _produces_metrics: list(str)
        The metrics that this node produces
    _selects_from: list(str)
        The status that this node selects from

    Methods
    -------
    __init__
    required_metrics
    required_global
    selects_from
    produces_global
    sets_status

    Notes
    -----
    Node specification is done by a dictionary, which specifies all of the
    parameters of the node. If the specification is not correct, you may
    get a runtime error.
    Specify the function with the name
        function: the function name to use
    There is currently only one function supported:
        metric_left_op_right
    with left the left-hand metric, op a binary comparison operator, and
    right the right-hand metric. You must specify which components may be
    operated on, and what the status will be set to if the condition is
    met. You may optionally scale the left and right hand sides.
    Required:
        left: left-hand metric
        right: right-hand metric
        op: a binary comparison operator as a string
        select: a list of statuses to be selected from the board
        set_true_status: the status to set if the result is True
    Optional:
        set_false_status: the status to set if the result is False
        scale_left: the scaling factor for the left metric
        scale_right: the scaling factor for the right metric
    For documentation purposes, any key name not relevant to the function
    will be ignored, so you may include names such as "_comment" in order
    to hack a comment if you generate this dictionary with a JSON.
    """
    def __init__(self, board, specification):
        if not isinstance(board, DecisionBoard):
            raise TypeError("Decision board must be a DecisionBoard")
        if not isinstance(specification, dict):
            raise TypeError("Nodes must be created from dicts")
        if not "function" in specification:
            raise ValueError("Specification dicts must have a function")
        self._board = board
        fn = specification["function"]
        if fn == "metric_left_op_right":
            self._build_metric_left_op_right(specification)
        else:
            raise ValueError(f"Function {fn} is not implemented")
    def _build_metric_left_op_right(self, specification):
        #TODO: think if there's a better way to do this
        required_keys = (
            "left", "right", "op", "select", "set_true_status"
        )
        for k in required_keys:
            if k not in specification:
                raise ValueError(f"metric_left_op_right requires {k}")
        # We know that globals will not be created
        self._produces_global = None
        self._required_global = None
        self._produces_metrics = []
        # Start pulling what we need
        left = specification["left"]
        right = specification["right"]
        self._required_metrics = [left, right]
        op = specification["op"]
        legal_ops = (">", ">=", "==", "<=", "<")
        if op not in legal_ops:
            raise ValueError(f"{op} is not a binary comparison operator")
        # op should be legal after this
        select = specification["select"]
        if not isinstance(select, list):
            select = [select]
        self._selects_from = select
        true_status = specification["set_true_status"]
        self._sets_status = true_status
        # handle optionals
        if "scale_left" in specification:
            scale_left = specification["scale_left"]
        else:
            scale_left = 1
        if "scale_right" in specification:
            scale_right = specification["scale_right"]
        else:
            scale_right = 1
        # We have everything now
        self._fn = lambda : self._metric_left_op_right(
            left,
            op,
            right,
            select,
            true_status,
            scale_left=scale_left,
            scale_right=scale_right,
        )
    def _metric_left_op_right(self, left, op, right, select, true_status, scale_left=1, scale_right=1):
        subtable = self._board.select([left, right], select)
        left_col = subtable[left]
        right_col = subtable[right]
        matches = eval(
            f"scale_left * left_col {op} scale_right * right_col"
        )
        comps = [c for c in subtable[matches]["Component"]]
        # set status for matches
        self._board.set_status(comps, true_status)
    def selects_statuses(self):
        """Returns the statuses this node selects

        Returns
        -------
        list(str) the list of statuses this node selects
        """
        return self._selects_from
    def sets_status(self):
        """Returns the status this node may set

        Returns
        -------
        str of the status this node may set
        """
        return self._sets_status
    def required_metrics(self):
        """Returns the metrics that this node requires

        Returns
        -------
        list(str) of the metrics required to run
        """
        return self._required_metrics
    def run(self):
        self._fn()
