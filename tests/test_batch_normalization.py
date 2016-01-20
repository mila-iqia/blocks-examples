from batch_normalization import main


def test_batch_normalization_example():
    bn_main_loop = main(num_epochs=2)
    assert bn_main_loop.log.current_row['test_misclass'] < 0.62
    assert bn_main_loop.log.current_row['train_misclass'] < 0.6
    nbn_main_loop = main(num_epochs=2, batch_normalized=False)
    assert nbn_main_loop.log.current_row['test_misclass'] > 0.65
    assert nbn_main_loop.log.current_row['train_misclass'] > 0.65
