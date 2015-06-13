Learning to approximate the square root function
================================================================

This is a super-basic example, mainly for testing purposes.

This script trains a tiny network to compute the square root of its input.

Running this example
--------------------------

Go into the REPO root directory (i.e. into the folder that *contains* the 
``sqrt`` directory) and run this module as follows::

    python -m sqrt --num-batches 1000 sqrt/saved_state


Structure of the Data
--------------------------

This example constructs a datastream on-the-fly, rather than
reading from disk.  


Structure of the Model
--------------------------

The model being used is a standard Multi-Layer Perceptron (``MLP``),
with 1 input node, 10 hidden nodes and 1 output node.

The (internal) outputs from the hidden nodes have ``Tanh`` non-linearities 
applied.  Then, these (bounded) values are linearly combined to sum 
to the output value, which in turn has an ``Identity`` 'non-linearity' applied - 
i.e. it is simply a linear combination of the values at the hidden layer.


Structure of the Training
--------------------------

The network learns from training examples in the ``range(0,100)``,
using regular ``GradientDescent`` on the ``SquaredError`` cost function.


Structure of the Testing
--------------------------

The network is periodically tested on examples in the ``range(100,200)`` -
so it shouldn't be surprising that the testing error isn't driven to 
zero with the training data, since the learning and testing 
are over different (and disjoint) ranges of inputs.

