import tempfile

from blocks.serialization import load

from mnist import main


def test_mnist():
    with tempfile.NamedTemporaryFile() as f:
        main(f.name, 1, True)
        with open(f.name, "rb") as source:
            main_loop = load(source)
        main_loop.find_extension("FinishAfter").set_conditions(
            after_n_epochs=2)
        main_loop.run()
        assert main_loop.log.status['epochs_done'] == 2
