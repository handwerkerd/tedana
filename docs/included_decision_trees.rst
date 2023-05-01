#######################
Included Decision Trees
#######################

Two decision trees are currently distributed with ``tedana``.
``kundu`` is the decision tree that was based on MEICA version 2.5
and has been included with ``tedana`` since when this project started.
Users have been generally content with the kundu tree, but it includes
many steps with arbitrary thresholds. ``minimal`` is a simplified 
version of that decision tree with fewer steps and arbitrary thresholds.
Minimal is designed to be more stable and comprehensible, but it has not
yet be extensively validated and parts of the tree may change in 
response to testing. 


Each tree takes as input a table with metrics, like :math:`\kappa` or
:math:`\rho`, for each component. Each step or node in the decision tree
either calculates new values or changes component classifications based on
these metrics.

Flowcharts describing the steps in both trees are below. Each step is labeled
with a ``node`` number. If ``tedana`` is run using one of these trees, those node
numbers will make the numbers in the ``ICA status table`` and the
``ICA decision tree`` that are `saved with the outputs`_. These flow charts
can help understand what happened in a step where a component's classifiation changed.

.. _saved with the outputs: output_file_descriptions.html

*******************
Kundu decision tree
*******************

Text n stuff

.. raw:: html

    <img src = "_static/kundu_decision_tree.svg" alt="Kundu Decision Tree Flow Chart"/>



*********************
Minimal decision tree
*********************

Text n stuff

.. raw:: html

    <img src = "_static/minimal_decision_tree.svg" alt="Minimal Decision Tree Flow Chart"/>
