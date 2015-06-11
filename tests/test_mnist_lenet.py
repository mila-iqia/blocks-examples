import tempfile
from blocks.serialization import load

from mnist_lenet import main


def test_mnist_lenet():
    with tempfile.NamedTemporaryFile() as f:
        main(f.name, 1)
        with open(f.name, "rb") as source:
            main_loop = load(source)
    main_loop.find_extension("FinishAfter").set_conditions(
        after_n_epochs=2)
    main_loop.run()
    assert main_loop.log.status['epochs_done'] == 2
