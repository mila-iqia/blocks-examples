"""This example shows how to train a simple RNN for the sequence classification
task: given a sequence of 0s and 1s, determine whether number of 1s in it
is odd or even
"""

import argparse
import logging

from parity_problem import main

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser("This example trains a simple LSTM "
                                     "network that determines if number of 1s "
                                     "in a given sequence of 0s and 1s is odd "
                                     "or even")
    parser.add_argument("--max-seq-length", type=int, default=100,
                        help="Max length of sequence to learn.")
    parser.add_argument("--lstm-dim", type=int, default=1,
                        help="Number of hidden units in the LSTM layer.")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Number of examples in a single batch.")
    parser.add_argument("--num-batches", type=int, default=1000,
                        help="Number of batches in the training dataset.")
    parser.add_argument("--num-epochs", type=int, default=30,
                        help="Number of epochs to do.")
    args = parser.parse_args()
    main(**vars(args))
