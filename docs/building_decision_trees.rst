########################################################
Understanding and building a component selection process
########################################################

``tedana`` involve transforming data into components via ICA, and then calculating metrics for each component.
Each metric has one value per component that is stored in a comptable or component_table dataframe. This structure
is then pass to a "decision tree" which through a series of binary choices categories each component as accepted or
rejected. The time series for the rejected components are regressed from the data in the final denoising step.

There are several decision trees that are included by default in ``tedana`` but users can also build their own.
This might be useful both if one of the default decision trees needs to be slightly altered due to the nature
of a specific data set, if one has an idea for a new approach to multi-echo denoising, or if one wants to integrate
non-multi-echo metrics into a single decision tree.

Note: We use two terminologies interchangably. This whole process is called "component selection"
and much of the code uses variants of that phrase (i.e. the ComponentSelector class, selection_nodes for the functions used in selection).
Instructions for how to classify components is called a "decision tree" since each step in the selection
process branches components into different intermediate or final classifications

.. contents:: :local:


******************************************
Expected outputs after component selection
******************************************

*New rows in the component_table or comptable*

The default file name for the component table is: ``desc-tedana_metrics.tsv``

classification:
    In the final table, the only values should be 'accepted' or 'rejected'.
    While the decision table is bring running, there may also be intermediate
    classification labels. Note, nothing in the current code requires a tree to
    assign one of these two labels to every component. There will be a warning
    if other labels remain

classification_tags:
    Human readable tags that explain why a classification was reached. These can
    be things like 'Likely BOLD', 'Unlikely BOLD', 'low variance' (i.e. accepted
    because the variance is too low to low a degree of freedom by calling it noise).
    Each component can have no tags (an empty string), one tag, or a comma separated
    list of tags.

**Data stored in the ComponentSelector object**

cross_component_metrics:
    Metrics that are each a single value calculated across components. For example, kappa and rho elbows.

component_status_table:
    A table where each column lists the classification status of
    each component after each node was run. This is useful for understanding the classification
    path of each component through the decision tree

used_metrics:
    A list of the metrics that were used in the decision tree.

classification_tags:
    A list of the pre-specified classification tags that could be used in a decision tree.
    Any reporting interface should use this field so that the tags that are possible are listed
    even if no components are assigned a specific tag.
    
**Outputs of each decision tree step**

This includes all the information from the inputted decision tree under each "node" or function
call. For each node, there is also an "outputs" subfield with information from when the tree
was executed
(Currently also in selector, but should be saved as a json file)

decison_node_idx:
    The decision tree functions are run as part of an ordered list.
    This is the positional index for when this function was run
    as part of this list. (First index is 0)
    
used_metrics:
    A list of the metrics used in a node of the decision tree

used_cross_component_metrics:
    A list of cross component metrics used in the node of a decision tree

node_label:
    A brief label for what happens in this node that can be used in a decision
    tree summary table or flow chart.

numTrue, numFalse:
    For decision tree (dec) functions, the number of components that were classified
    as true or false respectively in this decision tree step.

calc_cross_comp_metrics:
    For calculation (calc) functions, cross component metrics that were
    calculated in this function. When this is included, each of those
    metrics and the calculated values are also distinct keys in 'outputs'.
    While the cross component metrics table does not include where each component
    was calculated, that information is stored here.


*********************************
Defining a custom a decision tree
*********************************

Decision trees are stored in json files. The default trees are with the tedana code in ./resources/decision_trees
The minimal tree, minimal.json is a good example highlighting the structure and steps in a tree. It may be helpful
to look at that tree while reading this section.

A user can specify another decision tree and link to the tree location when tedana is executed. The format is
flexible to allow for future innovations, but that also means, it's flexible enough for someone who designs a tree
to create something with non-ideal results for the current code. Some criteria will result in an error
if violated, but more will just give a warning. If you are designing or editing a tree, look carefully at the warnings.

A decision tree can include two types of nodes or functions. All functions are currently in selection_nodes.py

- A decision function will use existing metrics and potentially change the classification of the components based on those metrics. By convention, all these functions should begin with "dec"
- A calculation function will take existing metrics and calculate a value across components to be used for classification, for example the kappa and rho elbows. By convention, all these functions should begin with "calc"
Nothing prevents a function from both calculating new cross component values and applying those values in a decision step, but following this convention should hopefully make decision tree specifications easier to follow and interpret.

**Key expectations**

- All trees should start with a "manual_classification" node that should set all component classifications to "unclassified" and
  have "clear_classification_tags" set to true. There might be special cases where someone might want to violate these rules
  but, depending what else happens in preceding code, other functions will expect both of these columns to exist.
  This manual_classification step will make sure those columns are created and initialized.
- Every possible path through the tree should result in each component being classified as 'accepted' or 'rejected'
- Three initialization variables will help prevent mistakes
  
  necessary_metrics:
      Is a list of the necessary metrics in the component table that will be used by the tree. If a metric doesn't exist then this
      will raise an error instead of executing a tree. (This can eventually be used to call the metric calculation code based on
      the decision tree specification). If a necessary metric isn't used, there will be a warning. This is just a warning because,
      if the decision tree code specification is eventually used to calculated metrics, one may want to calculate a metric even if
      it's not being used.

  intermediate_classifications:
      A list of intermediate classifications (i.e. "provisionalaccept", "provisionalreject"). It is very important to prespecify these
      because the code will make sure only the default classifications ("accepted" "rejected" "unclassified") and intermediate classifications
      are used in a tree. This prevents someone from accidentially losing a component due to a spelling error or other minor variation in a
      classification label

  classification_tags:
      A list of acceptable classification tags (i.e. "Likely BOLD", "Unlikely BOLD", "Low variance"). This will both be used to make sure only
      these tags are used in the tree and allow programs that interact with the results one place to see all potential tags

**Decision node json structure**

There are  6 initial fields, necessary_metrics, intermediate_classification, and classification_tags, as described in the above section:

- "tree_id": a descriptive name for the tree that will be logged.
- "info": A brief description of the tree for info logging
- "report": A narrative description of the tree that could be used in report logging
- "refs" Publications that should be referenced, when this tree is used

The "nodes" field is a list of elements where each element defines a node the decision tree. There are several key fields for each of these nodes:

- "functionname": The exact function name in selection_nodes.py that will be called.
- "parameters": Specifications of all required parameters for the function in functionname
- "kwargs": Specification for optional parameters for the function in functionname

The only parameter that is used in all functions is "decidecomps" which is used to identify, based on their classifications,
the components a function should be applied to. It can be a single classification, or a comma separated string of classificaions.
In addition to the intermediate and default ("accepted" "rejected" "unclassified") component classifications, this can be "all"
for functions that should be applied to all components regardless of their classifications

Most decision functions also include "ifTrue" and "ifFalse" which specify how to change the classification of each component
based on whether a the decision criterion is true or also. In addition to the default and intermediate classification options,
this can also be "nochange" (i.e. For components where a>b is true, "reject". For components where a>b is false, "nochange").
The optional parameters "tag_ifTrue" and "tag_ifFalse" define the classification tags to be assigned to components.
Currently, the only exception is "manual_classify" which uses "new_classification" to designate the new component classification
and "tag" (optional) to designate which classification tag to apply.

There are several optional parameters in every decision tree function:

- custom_node_label: A brief label for what happens in this node that can be used in a decision tree summary table or flow chart. If custom_node_label is not not defined, then each function has default descriptive text.
- log_extra_report, log_extra_info: Text for each function call is automatically placed in the logger output. In addition to that text, the text in these these strings will also be included in the logger with the report or info codes respectively. These might be useful to give a narrative explanation of why a step was parameterized a certain way.
- only_used_metrics: If true, this function will only return the names of the component table metrics that will be used when this function is fully run. This can be used to identify all used metrics before running the decision tree.

********************************
Key parts of selection functions
********************************

There are several expectations for selection functions that are necessary for them to properly execute.
In selection_nodes.py, manual_classify, dec_left_op_right, and calc_kappa_rho_elbows_kundu are good
examples for how to meet these expectations.

Create a dictionary called "outputs" that includes key fields that should be recorded. 
The following line should be at the end of each function ``selector.nodes[selector.current_node_idx]["outputs"] = outputs`` 
Additional fields can be used to log funciton-specific information, but the following fields are common and may be used by other parts of the code:

- "decision_node_idx" (required): the ordered index for the current function in the decision tree.
- "node_label" (required): A decriptive label for what happens in the node.
- "numTrue" & "numFalse" (required for decision functions): For decision functions, the number of components labels true or false within the function call.
- "used_metrics" (required if a function uses metrics): The list of metrics used in the function. This can be hard coded, defined by input parameters, or empty.
- "used_cross_component_metrics" (required if a function uses cross component metrics): A list of cross component metrics used in the function. This can be hard coded, defined by input parameters, or empty.
- "calc_cross_comp_metrics" (required for calculation functions): A list of cross component metrics calculated within the function. The key-value pair for each calculated metric is also included in "outputs"

Before anything data are touched in the function, there should be an ``if only_used_metrics:`` clause that returns ``used_metrics`` for the function call

Existing functions define ``function_name_idx = f"Step {selector.current_node_idx}: [text of function_name]`` This is used several times in logging and is nice to define only once.


Code the executes ``outputs["node_label"] = custom_node_label`` if there is a user-inputted custom node label or assigned a default node label. The default node lable
may be used in decision tree visualization so it should be relatively short.

Calculation nodes should check if the value they are calculating was already calculated and output a warning if the function overwrites and existing value

Code that adds the text log_extra_info and log_extra_report into the appropriate logs (if they are provided by the user)

After the above information is included, all functions will call ``selectcomps2use`` which returns the components with classifications included in ``decide_comps``
Then run ``confirm_metrics_exist`` which is an added check to make sure the metrics used by this function exist in the component table.

Nearly every function has a clause like:

.. code-block:: python

  if comps2use is None:
     log_decision_tree_step(function_name_idx, comps2use, decide_comps=decide_comps)
     outputs["numTrue"] = 0
     outputs["numFalse"] = 0
  else:

If there are no components with the classifications in ``decide_comps`` this logs that there's nothing for the function to be run on, else continue.

For decision functions the key variable is ``decision_boolean`` which should be a dataframe column which is True or False based on the function's criteria.
That column is an input to ``change_comptable_classifications`` which will update the component_table classifications, update the classification history in component_status_table,
and update the component classification_tags.

This is followed by something that logs how many components were identified as true or false, like:

.. code-block:: python

  outputs["numTrue"] = np.asarray(decision_boolean).sum()
  outputs["numFalse"] = np.logical_not(decision_boolean).sum()

For calculation functions, the calculated values should be added as a value/key pair to both ``selector.cross_component_metrics`` and ``outputs``

``log_decision_tree_step`` puts the relevant info from the function call into the program's output log.

Every function should end.

.. code-block:: python

      selector.nodes[selector.current_node_idx]["outputs"] = outputs
      return selector

  functionname.__doc__ = (functionname.__doc__.format(**decision_docs))

This returns makes sure the outputs from the function are saved in the class structure and the class structure is return.
The following line should include the function's name and is used to make sure repeated variable names are compiled correctly for the API documentation.

If you follow these simple steps you'll be able design your very own decision tree functions.
