import tempfile

from six.moves import cPickle

from mnist_lenet import main


def test_mnist_lenet():
    with tempfile.NamedTemporaryFile() as f:
        main(f.name, 1, True)
        with open(f.name, "rb") as source:
            main_loop = cPickle.load(source)
    main_loop.find_extension("FinishAfter").set_conditions(
        after_n_epochs=2)
    main_loop.run()
    assert main_loop.log.status['epochs_done'] == 2
