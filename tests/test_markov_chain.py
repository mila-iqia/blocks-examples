import tempfile

from markov_chain.main import main


def test_markov_chain():
    with tempfile.NamedTemporaryFile() as f:
        main("train", f.name, None, 10)
