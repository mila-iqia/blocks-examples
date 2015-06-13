"""Super-basic example, mainly for testing purposes.

This script trains a tiny network to compute square roots.

"""

import argparse
import logging

from sqrt import main

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser("An example of learning to calcuate square roots")
    parser.add_argument("--num-batches", type=int, default=1000,
                        help="Number of training batches to do.")
    parser.add_argument("save_to", default="sqrt", nargs="?",
                        help=("Destination path to save the state of the training "
                              "process."))
    args = parser.parse_args()
    main(**vars(args))

