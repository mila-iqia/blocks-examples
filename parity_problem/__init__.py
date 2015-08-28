from __future__ import print_function

import random
import numpy as np
import theano
import theano.tensor as T

from fuel.streams import DataStream
from fuel.datasets import IterableDataset

from blocks.bricks import Linear, Logistic
from blocks.bricks.recurrent import LSTM
from blocks.bricks.cost import BinaryCrossEntropy
from blocks.initialization import Constant, IsotropicGaussian
from blocks.algorithms import GradientDescent, Adam
from blocks.graph import ComputationGraph
from blocks.extensions import FinishAfter, Printing, ProgressBar
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.main_loop import MainLoop


def generate_data(max_seq_length, batch_size, num_batches):
    x = []
    y = []
    for i in range(num_batches):
        # it's important to include sequences of different length in
        # the training data in order the model was able to learn something
        seq_length = random.randint(1, max_seq_length)

        # each batch consists of sequences of equal length
        # x_batch has shape (T, B, F=1)
        x_batch = np.random.random_integers(0, 1, (seq_length, batch_size, 1))
        # y_batch has shape (B, F=1)
        y_batch = x_batch.sum(axis=(0,)) % 2
        x.append(x_batch.astype(theano.config.floatX))
        y.append(y_batch.astype(theano.config.floatX))
    return {'x': x, 'y': y}


def main(max_seq_length, lstm_dim, batch_size, num_batches, num_epochs):
    dataset_train = IterableDataset(generate_data(max_seq_length, batch_size,
                                                  num_batches))
    dataset_test = IterableDataset(generate_data(max_seq_length, batch_size,
                                                 100))

    stream_train = DataStream(dataset=dataset_train)
    stream_test = DataStream(dataset=dataset_test)

    x = T.tensor3('x')
    y = T.matrix('y')

    # we need to provide data for the LSTM layer of size 4 * ltsm_dim, see
    # LSTM layer documentation for the explanation
    x_to_h = Linear(1, lstm_dim * 4, name='x_to_h',
                    weights_init=IsotropicGaussian(),
                    biases_init=Constant(0.0))
    lstm = LSTM(lstm_dim, name='lstm',
                weights_init=IsotropicGaussian(),
                biases_init=Constant(0.0))
    h_to_o = Linear(lstm_dim, 1, name='h_to_o',
                    weights_init=IsotropicGaussian(),
                    biases_init=Constant(0.0))

    x_transform = x_to_h.apply(x)
    h, c = lstm.apply(x_transform)

    # only values of hidden units of the last timeframe are used for
    # the classification
    y_hat = h_to_o.apply(h[-1])
    y_hat = Logistic().apply(y_hat)

    cost = BinaryCrossEntropy().apply(y, y_hat)
    cost.name = 'cost'

    lstm.initialize()
    x_to_h.initialize()
    h_to_o.initialize()

    cg = ComputationGraph(cost)

    algorithm = GradientDescent(cost=cost, parameters=cg.parameters,
                                step_rule=Adam())
    test_monitor = DataStreamMonitoring(variables=[cost],
                                        data_stream=stream_test, prefix="test")
    train_monitor = TrainingDataMonitoring(variables=[cost], prefix="train",
                                           after_epoch=True)

    main_loop = MainLoop(algorithm, stream_train,
                         extensions=[test_monitor, train_monitor,
                                     FinishAfter(after_n_epochs=num_epochs),
                                     Printing(), ProgressBar()])
    main_loop.run()

    print('Learned weights:')
    for layer in (x_to_h, lstm, h_to_o):
        print("Layer '%s':" % layer.name)
        for param in layer.parameters:
            print(param.name, ': ', param.get_value())
        print()

    return main_loop
