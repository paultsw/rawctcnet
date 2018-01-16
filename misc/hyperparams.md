Hyperparameter Search Methodology
=================================

Focus
-----
We focus on tuning the following settings during hyperparameter search:
* convolutional layers;
* padding;
* BatchNorm;
* learning rate and gradient descent algorithms;
* batch sizes.

Elaboration on each point follows.

Convolutional Layers
--------------------
What are the optimal dilation rates and number of residual stacks for convergence/loss?

Padding
-------
Does padding (...)

BatchNorm
---------
Does adding BatchNorm before each residual convolutional block assist with improved learning and convergence?

Learning Rates
--------------

Batch Size
----------

Tests
-----
We will run the following hyperparameter-tuning tests; all are with respect to a single epoch on a tiny dataset.

* (TBD)
* (TBD)
* (TBD)
* (TBD)
