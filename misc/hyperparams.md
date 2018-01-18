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

We test this simultaneously with batch sizes; this is due to the dependency between the two settings. Generally,
we scale the learning rate by a factor of `sqrt(C)`, where `C` is the factor by which we scale the batch size.

Batch Size
----------
Does an increased batch size offer improved convergence and learning?

We try batch sizes in `1,4,8,16,32`.

We test this simultaneously with learning rates; this is due to the dependency between the two settings.

Roughly, we take the best optimization algorithm from above and scale the learning rate by the square root of the
scaling factor of the batch size increase/decrease. We also look at a slightly higher/lower learning rate (2x/0.5x)
just in case.

Convolutional Layers
--------------------
What are the optimal dilation rates and number of residual stacks for accuracy?

We vary over two ranges of settings:
* `MAX_DIL`: maximum dilation within each residual stack (the biggest power of 2) -- `16,32,64,128`;
* `NUM_STACKS`: number of residual stacks of dilated convolutions: `3,6`.

We also note that as the model size increases, smaller and smaller learning rates should be used. This is due to
the increase in non-convexity (e.g. number of "hills and valleys") in the error surface corresponding to the parameter
space. In the tests below, we will start off with the optimal learning rate and batch size as discovered in earlier
tests, and decrease the learning rate by a factor of 1/2 four times and note the best result.

Tests
-----
We will run the following hyperparameter-tuning tests; all are with respect to a single epoch on a tiny dataset.

In the following, `<OPT>` means the optimal setting for that parameter, discovered empirically in the tests prior to that one.

#### Padding:
* [X] `PADDING=OFF`, `BATCH=8`, `BATCHNORM=FALSE`, `MAX_DIL=64`, `NUM_STACK=3`, `LR=ADAM@0.0001`
* [X] `PADDING=ON`, `BATCH=8`, `BATCHNORM=FALSE`, `MAX_DIL=64`, `NUM_STACK=3`, `LR=ADAM@0.0001`
* _Conclusion: padding provides little-to-no benefit. Turn it off._
#### Batch-Norm:
* [X] `PADDING=<OPT>`, `BATCH=8`, `BATCHNORM=TRUE`, `MAX_DIL=64`, `NUM_STACK=3`, `LR=ADAM@0.0001`
* _Conclusion: batch-norm provides a huge increase in convergence speed during training. Keep it on._
#### LR (AdamOpt):
* [X] `PADDING=<OPT>`, `BATCH=8`, `BATCHNORM=<OPT>`, `MAX_DIL=64`, `NUM_STACK=3`, `LR=ADAM@0.001` (_Val.loss: 41.688525286587804_)
* [X] `PADDING=<OPT>`, `BATCH=8`, `BATCHNORM=<OPT>`, `MAX_DIL=64`, `NUM_STACK=3`, `LR=ADAM@0.0005` (_Val.loss: 34.19132614135742_)
* [X] `PADDING=<OPT>`, `BATCH=8`, `BATCHNORM=<OPT>`, `MAX_DIL=64`, `NUM_STACK=3`, `LR=ADAM@0.0001` (_Val.loss: 11.2378380948847_)
* [X] `PADDING=<OPT>`, `BATCH=8`, `BATCHNORM=<OPT>`, `MAX_DIL=64`, `NUM_STACK=3`, `LR=ADAM@0.00005` (_Val.loss: 10.11766067418185_)
* _Conclusion: comparable performance to Adamax._
#### LR (AdaMaxOpt):
* [X] `PADDING=<OPT>`, `BATCH=8`, `BATCHNORM=<OPT>`, `MAX_DIL=64`, `NUM_STACK=3`, `LR=ADAMAX@0.002` (_Val.loss: 155.1495721990412_)
* [X] `PADDING=<OPT>`, `BATCH=8`, `BATCHNORM=<OPT>`, `MAX_DIL=64`, `NUM_STACK=3`, `LR=ADAMAX@0.001` (_Val.loss: 52.05568382956765_)
* [X] `PADDING=<OPT>`, `BATCH=8`, `BATCHNORM=<OPT>`, `MAX_DIL=64`, `NUM_STACK=3`, `LR=ADAMAX@0.0005` (_Val.loss: 8.574921954761852_)
* [X] `PADDING=<OPT>`, `BATCH=8`, `BATCHNORM=<OPT>`, `MAX_DIL=64`, `NUM_STACK=3`, `LR=ADAMAX@0.0001` (_Val.loss: 14.724966699426824_)
* _Conclusion: 0.0005 gave the best convergence._
#### LR (AdagradOpt):
* [X] `PADDING=<OPT>`, `BATCH=8`, `BATCHNORM=<OPT>`, `MAX_DIL=64`, `NUM_STACK=3`, `LR=ADAGRAD@0.01` (_Val.loss: 26.675187717784535_)
* [X] `PADDING=<OPT>`, `BATCH=8`, `BATCHNORM=<OPT>`, `MAX_DIL=64`, `NUM_STACK=3`, `LR=ADAGRAD@0.005` (_Val.loss: 14.264448079195889_)
* [X] `PADDING=<OPT>`, `BATCH=8`, `BATCHNORM=<OPT>`, `MAX_DIL=64`, `NUM_STACK=3`, `LR=ADAGRAD@0.001` (_Val.loss: 11.691955696452748_)
* [X] `PADDING=<OPT>`, `BATCH=8`, `BATCHNORM=<OPT>`, `MAX_DIL=64`, `NUM_STACK=3`, `LR=ADAGRAD@0.0005`(_Val.loss: 16.059732740575615_)
* _Conclusion: performance at lower learning rates is not as good as Adam/Adamax._
#### Batch Size:
* [X] `PADDING=<OPT>`, `BATCH=1`, `BATCHNORM=<OPT>`, `MAX_DIL=64`, `NUM_STACK=3`, `LR=<OPT>@<OPT>*0.5` (_Val.loss: 1.0764162296598607_)
* [X] `PADDING=<OPT>`, `BATCH=4`, `BATCHNORM=<OPT>`, `MAX_DIL=64`, `NUM_STACK=3`, `LR=<OPT>@<OPT>*0.7` (_Val.loss: 4.33007783239538_)
* [X] `PADDING=<OPT>`, `BATCH=8`, `BATCHNORM=<OPT>`, `MAX_DIL=64`, `NUM_STACK=3`, `LR=<OPT>@<OPT>*1.0` (_Val.loss: 10.196995084935969_)
* [X] `PADDING=<OPT>`, `BATCH=16`, `BATCHNORM=<OPT>`, `MAX_DIL=64`, `NUM_STACK=3`, `LR=<OPT>@<OPT>*1.4` (_Val.loss: 15.626627011732621_)
* [X] `PADDING=<OPT>`, `BATCH=32`, `BATCHNORM=<OPT>`, `MAX_DIL=64`, `NUM_STACK=3`, `LR=<OPT>@<OPT>*2.0` (_Val.loss: 56.45656065507369_)
* _Conclusion: We chose AdaMax@LR=0.0005 as the default optimizer and learning rate. We noticed that there was a trade-off between batch size and accuracy on validation; this leads us to a strategy in which we can consider training on large batches at the start and fine-tuning on smaller batch-sizes afterwards, as large batch sizes lead to faster convergence at the tradeoff of a worse validation loss._
#### Convolutional Layers (Dilation):
* [ ] `PADDING=<OPT>`, `BATCH=<OPT>`, `BATCHNORM=<OPT>`, `MAX_DIL=16`, `NUM_STACK=3`, `LR=<OPT>`
* [ ] `PADDING=<OPT>`, `BATCH=<OPT>`, `BATCHNORM=<OPT>`, `MAX_DIL=32`, `NUM_STACK=3`, `LR=<OPT>`
* [ ] `PADDING=<OPT>`, `BATCH=<OPT>`, `BATCHNORM=<OPT>`, `MAX_DIL=64`, `NUM_STACK=3`, `LR=<OPT>`
* [ ] `PADDING=<OPT>`, `BATCH=<OPT>`, `BATCHNORM=<OPT>`, `MAX_DIL=128`, `NUM_STACK=3`, `LR=<OPT>`
* _Conclusion: (...)_
#### Convolutional Layers (Stacks)
* [ ] `PADDING=<OPT>`, `BATCH=<OPT>`, `BATCHNORM=<OPT>`, `MAX_DIL=<OPT>`, `NUM_STACK=6`, `LR=<OPT>`
* [ ] `PADDING=<OPT>`, `BATCH=<OPT>`, `BATCHNORM=<OPT>`, `MAX_DIL=<OPT>`, `NUM_STACK=9`, `LR=<OPT>`
* _Conclusion: (...)_
