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

Padding
-------
Does padding affect the accuracy of the output when batching occurs? A concern is that dynamic batch-padding will cause
the network to believe that trailing zeros from padding are relevant (though the risk is minimal due to the structure of
the CTC loss, and the fact that we can mask them out during validation). This is more of a sanity check than anything.

We test on a batch size of 8, on both padded and unpadded `signals.data.pth` datasets.

We do this before all other tests.

BatchNorm
---------
Does adding BatchNorm before each residual convolutional block assist with improved learning and convergence?

We examine convergence speed both with and without batch norm on a batch size of 8 and fixed `MAX_DIL=64, NUM_STACKS=3`.

Learning Rates
--------------
Which learning rate/SGD algorithm offers optimal convergence?

We try `Adam`, `Adamax`, and `Adagrad` with varying learning rates, decreasing from the default values of
`lr=0.001, lr=0.002, lr=0.01` (respectively) by a factor 2-5, with 4 decreases each.

The best learning rate is dependent on the number of convolutional layers, but we will make two assumptions:
that the best optimization algorithm will not vary based on the number of convolutional layers (probably reasonable),
and that an increase in the number of layers will require a roughly proportionate decrease in the learning rate
(this will need to be empirically tested with another round of learning rate hyperparameter search after choosing
a model size).

We test this simultaneously with batch sizes; this is due to the dependency between the two settings.

Batch Size
----------
Does an increased batch size offer improved convergence and learning?

We try batch sizes in `1,4,8,16,32`.

We test this simultaneously with learning rates; this is due to the dependency between the two settings.

Convolutional Layers
--------------------
What are the optimal dilation rates and number of residual stacks for accuracy?

We vary over two ranges of settings:
* `MAX_DIL`: maximum dilation within each residual stack (the biggest power of 2) -- `16,32,64,128`;
* `NUM_STACKS`: number of residual stacks of dilated convolutions: `3,6`.

Tests
-----
We will run the following hyperparameter-tuning tests; all are with respect to a single epoch on a tiny dataset.

In the following, `<OPT>` means the optimal setting for that parameter, discovered empirically in the tests prior to that one.

* `PADDING=OFF`, `BATCH=8`, `BATCHNORM=FALSE`, `MAX_DIL=64`, `NUM_STACK=3`, `LR=ADAM@0.0001`.
* `PADDING=ON`, `BATCH=8`, `BATCHNORM=FALSE`, `MAX_DIL=64`, `NUM_STACK=3`, `LR=ADAM@0.0001`.
* `PADDING=<OPT>`, `BATCH=8`, `BATCHNORM=TRUE`, `MAX_DIL=64`, `NUM_STACK=3`, `LR=ADAM@0.0001`.
* (LR and batch size tests TBD)
* (Convolutional layer tests TBD)
