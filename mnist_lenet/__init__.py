import logging
from argparse import ArgumentParser
import numpy

from theano import tensor

from blocks.algorithms import GradientDescent, Scale
from blocks.bricks import MLP, Rectifier, Initializable, FeedforwardSequence, Softmax
from blocks.bricks.conv import (
    ConvolutionalLayer, ConvolutionalSequence, Flattener)
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.initialization import IsotropicGaussian, Constant
from fuel.streams import DataStream
from fuel.datasets import MNIST
from fuel.schemes import SequentialScheme
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_dropout
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.extensions.plot import Plot
from blocks.main_loop import MainLoop
from blocks.roles import INPUT
from blocks.utils import named_copy


class LeNet(FeedforwardSequence, Initializable):
    def __init__(self, conv_activations, num_channels, image_shape,
                 filter_sizes, feature_maps, pooling_sizes,
                 top_mlp_activations, top_mlp_dims, conv_step=None,
                 border_mode='valid', **kwargs):
        if conv_step is None:
            self.conv_step = (1, 1)
        else:
            self.conv_step = conv_step
        self.num_channels = num_channels
        self.image_shape = image_shape
        self.top_mlp_activations = top_mlp_activations
        self.top_mlp_dims = top_mlp_dims
        self.border_mode = border_mode

        params = zip(conv_activations, filter_sizes, feature_maps,
                     pooling_sizes)
        self.layers = [ConvolutionalLayer(filter_size=filter_size,
                                          num_filters=num_filter,
                                          pooling_size=pooling_size,
                                          activation=activation.apply,
                                          conv_step=self.conv_step,
                                          border_mode=self.border_mode,
                                          name='conv_pool_{}'.format(i))
                       for i, (activation, filter_size, num_filter,
                               pooling_size)
                       in enumerate(params)]
        self.conv_sequence = ConvolutionalSequence(self.layers, num_channels,
                                                   image_size=image_shape)

        application_methods = [self.conv_sequence.apply]
        self.top_mlp = MLP(top_mlp_activations, top_mlp_dims)
        self.flattener = Flattener()
        if len(top_mlp_activations) > 0:
            application_methods += [self.flattener.apply]
            application_methods += [self.top_mlp.apply]
        super(LeNet, self).__init__(application_methods, **kwargs)

    @property
    def output_dim(self):
        return self.top_mlp_dims[-1]

    @output_dim.setter
    def output_dim(self, value):
        self.top_mlp_dims[-1] = value

    def _push_allocation_config(self):
        self.conv_sequence._push_allocation_config()
        conv_out_dim = self.conv_sequence.get_dim('output')

        self.top_mlp.activations = self.top_mlp_activations
        self.top_mlp.dims = [numpy.prod(conv_out_dim)] + self.top_mlp_dims


def main(save_to, num_epochs, bokeh=False, feature_maps=None,
         mlp_hiddens=None, conv_sizes=None, pool_sizes=None):
    if feature_maps is None:
        feature_maps = [4, 6]
    if mlp_hiddens is None:
        mlp_hiddens = [500]
    if conv_sizes is None:
        conv_sizes = [5, 5]
    if pool_sizes is None:
        pool_sizes = [2, 2]
    conv_activations = [Rectifier() for _ in feature_maps]
    mlp_activations = [Rectifier() for _ in mlp_hiddens] + [Softmax()]
    convnet = LeNet(conv_activations, 1, (28, 28),
                    filter_sizes=zip(conv_sizes, conv_sizes),
                    feature_maps=feature_maps,
                    pooling_sizes=zip(pool_sizes, pool_sizes),
                    top_mlp_activations=mlp_activations,
                    top_mlp_dims=mlp_hiddens + [10],
                    border_mode='full',
                    weights_init=IsotropicGaussian(0.1),
                    biases_init=Constant(0))
    convnet.initialize()

    x = tensor.tensor3('features')
    y = tensor.lmatrix('targets')
    probs = convnet.apply(x.dimshuffle(0, 'x', 1, 2))
    cost = named_copy(CategoricalCrossEntropy().apply(y.flatten(),
                      probs), 'cost')
    error_rate = named_copy(MisclassificationRate().apply(y.flatten(), probs),
                            'error_rate')

    cg = ComputationGraph([cost, error_rate])
    vars_for_dropout = VariableFilter(
        roles=[INPUT], bricks=[convnet.top_mlp])(cg.variables)
    cg_train = apply_dropout(cg, vars_for_dropout, 0.5)
    train_cost, train_error_rate = cg_train.outputs

    mnist_train = MNIST("train", flatten=False)
    mnist_train_stream = DataStream(dataset=mnist_train,
                                    iteration_scheme=SequentialScheme(
                                        mnist_train.num_examples, 100))
    mnist_test = MNIST("test", flatten=False)
    mnist_test_stream = DataStream(dataset=mnist_test,
                                   iteration_scheme=SequentialScheme(
                                       mnist_test.num_examples, 100))

    algorithm = GradientDescent(
        cost=train_cost, params=cg_train.parameters,
        step_rule=Scale(learning_rate=0.01))
    extensions = [Timing(),
                  FinishAfter(after_n_epochs=num_epochs),
                  DataStreamMonitoring(
                      [cost, error_rate],
                      mnist_test_stream,
                      prefix="test"),
                  TrainingDataMonitoring(
                      [train_cost, train_error_rate,
                       aggregation.mean(algorithm.total_gradient_norm)],
                      prefix="train",
                      after_epoch=True),
                  Checkpoint(save_to),
                  Printing()]

    if bokeh:
        extensions.append(Plot(
            'MNIST LeNet example',
            channels=[
                ['test_final_cost',
                 'test_misclassificationrate_apply_error_rate'],
                ['train_total_gradient_norm']]))
    model = Model(cost)

    main_loop = MainLoop(
        algorithm,
        mnist_train_stream,
        model=model,
        extensions=extensions)

    main_loop.run()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser("An example of training an MLP on"
                            " the MNIST dataset.")
    parser.add_argument("--num-epochs", type=int, default=2,
                        help="Number of training epochs to do.")
    parser.add_argument("save_to", default="mnist.pkl", nargs="?",
                        help=("Destination to save the state of the training "
                              "process."))
    parser.add_argument("--bokeh", action='store_true',
                        help="Set if you want to use Bokeh ")
    parser.add_argument("--feature-maps", type=int, default=None)
    parser.add_argument("--mlp-hiddens", type=int, default=None)
    parser.add_argument("--conv-sizes", type=int, default=None)
    parser.add_argument("--pool-sizes", type=int, default=None)
    args = parser.parse_args()
    main(args.save_to, args.num_epochs, args.bokeh)

