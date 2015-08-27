from parity_problem import main


def test_parity_problem():
    main_loop = main(20, 1, 10, 10, 1)
    assert main_loop.log.status['epochs_done'] == 1
    assert main_loop.log.status['iterations_done'] == 10
