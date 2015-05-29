import tempfile

from blocks.extensions.saveload import SAVED_TO

from sqrt import main


def test_sqrt():
    save_path = tempfile.mktemp()
    main_loop = main(save_path, 7)
    assert main_loop.log[7][SAVED_TO][0] == save_path
