"""Example of batch normalization tackling a difficult optimization.

Running with --no-batch-normalize, we see that the algorithm makes very
very slow progress in the same amount of time.

"""
import argparse
from theano import tensor
from blocks.algorithms import Adam, GradientDescent
from blocks.bricks import MLP, BatchNormalizedMLP, Logistic, Softmax
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.extensions import Printing
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions import FinishAfter
from blocks.graph import (ComputationGraph, apply_batch_normalization,
                          get_batch_normalization_updates)
from blocks.initialization import Constant, IsotropicGaussian
from blocks.main_loop import MainLoop
from fuel.datasets.toy import Spiral
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme, SequentialScheme


def main(num_epochs=50, batch_normalized=True, alpha=0.1):
    """Run the example.

    Parameters
    ----------
    num_epochs : int, optional
        Number of epochs for which to train.

    batch_normalized : bool, optional
        Batch-normalize the training graph. Defaults to `True`.

    alpha : float, optional
        Weight to apply to a new sample when calculating running
        averages for population statistics (1 - alpha weight is
        given to the existing average).

    """
    if batch_normalized:
        # Add an extra keyword argument that only BatchNormalizedMLP takes,
        # in order to speed things up at the cost of a bit of extra memory.
        mlp_class = BatchNormalizedMLP
        extra_kwargs = {'conserve_memory': False}
    else:
        mlp_class = MLP
        extra_kwargs = {}
    mlp = mlp_class([Logistic(), Logistic(), Logistic(), Softmax()],
                    [2, 5, 5, 5, 3],
                    weights_init=IsotropicGaussian(0.2),
                    biases_init=Constant(0.), **extra_kwargs)
    mlp.initialize()

    # Generate a dataset with 3 spiral arms, using 8000 examples for
    # training and 2000 for testing.
    dataset = Spiral(num_examples=10000, classes=3,
                     sources=['features', 'label'],
                     noise=0.05)
    train_stream = DataStream(dataset,
                              iteration_scheme=ShuffledScheme(examples=8000,
                                                              batch_size=20))
    test_stream = DataStream(dataset,
                             iteration_scheme=SequentialScheme(
                                 examples=list(range(8000, 10000)),
                                 batch_size=2000))

    # Build a cost graph; this contains BatchNormalization bricks that will
    # by default run in inference mode.
    features = tensor.matrix('features')
    label = tensor.lvector('label')
    prediction = mlp.apply(features)
    cost = CategoricalCrossEntropy().apply(label, prediction)
    misclass = MisclassificationRate().apply(label, prediction)
    misclass.name = 'misclass'  # The default name for this is annoyingly long
    original_cg = ComputationGraph([cost, misclass])

    if batch_normalized:
        cg = apply_batch_normalization(original_cg)
        # Add updates for population parameters
        pop_updates = get_batch_normalization_updates(cg)
        extra_updates = [(p, m * alpha + p * (1 - alpha))
                         for p, m in pop_updates]
    else:
        cg = original_cg
        extra_updates = []

    algorithm = GradientDescent(step_rule=Adam(0.001),
                                cost=cg.outputs[0],
                                parameters=cg.parameters)
    algorithm.add_updates(extra_updates)

    main_loop = MainLoop(algorithm=algorithm,
                         data_stream=train_stream,
                         # Use the original cost and misclass variables so
                         # that we monitor the (original) inference-mode graph.
                         extensions=[DataStreamMonitoring([cost, misclass],
                                                          train_stream,
                                                          prefix='train'),
                                     DataStreamMonitoring([cost, misclass],
                                                          test_stream,
                                                          prefix='test'),
                                     Printing(),
                                     FinishAfter(after_n_epochs=num_epochs)])
    main_loop.run()
    return main_loop


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch normalization demo.")
    parser.add_argument('--num-epochs', '-n', default=50, type=int,
                        help='Number of epochs for which to train.')
    parser.add_argument('--no-batch-normalize', '-N', action='store_const',
                        const=True, default=False,
                        help='Turn off batch normalization, to see the '
                             'difference it makes.')
    parser.add_argument('--alpha', '-A', action='store',
                        type=float, default=0.05,
                        help='Moving average coefficient for population '
                             'statistics')
    args = parser.parse_args()
    main(args.num_epochs, not args.no_batch_normalize, args.alpha)
