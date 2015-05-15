from __future__ import print_function
import tempfile

from blocks.config import config

from reverse_words import main


def test_reverse_words():
    old_limit = config.recursion_limit
    config.recursion_limit = 100000
    with tempfile.NamedTemporaryFile() as f_save,\
            tempfile.NamedTemporaryFile() as f_data:
        with open(f_data.name, 'wt') as data:
            for i in range(10):
                print("A line.", file=data)
        main("train", f_save.name, 1, [f_data.name])
    config.recursion_limit = old_limit
