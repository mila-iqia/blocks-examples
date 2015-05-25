def get_config_cs2en():
    config = {}

    # Model related
    config['seq_len'] = 50
    config['enc_nhids'] = 1000
    config['dec_nhids'] = 1000
    config['enc_embed'] = 620
    config['dec_embed'] = 620
    config['saveto'] = 'search_model_cs2en'

    # Optimization related
    config['batch_size'] = 80
    config['sort_k_batches'] = 12
    config['step_rule'] = 'AdaDelta'
    config['step_clipping'] = 10
    config['weight_scale'] = 0.01

    # Regularization related
    config['weight_noise_ff'] = False
    config['weight_noise_rec'] = False
    config['dropout'] = 1.0

    # Vocabulary/dataset related
    datadir = '/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/'
    config['stream'] = 'cs-en'
    config['src_vocab'] = datadir + 'vocab.cs.pkl'
    config['trg_vocab'] = datadir + 'vocab.en.pkl'
    config['src_data'] = datadir + 'all.tok.clean.shuf.cs-en.cs'
    config['trg_data'] = datadir + 'all.tok.clean.shuf.cs-en.en'
    config['src_vocab_size'] = 40000
    config['trg_vocab_size'] = 40000
    config['unk_id'] = 1

    # Early stopping based on bleu related
    config['normalized_bleu'] = True
    config['bleu_script'] = datadir + 'multi-bleu.perl'
    config['val_set'] = datadir + '/newsdev2015.tok.seg.fi'
    config['val_set_grndtruth'] = data + '/newsdev2015_1.tok.en'
    config['val_set_out'] = config['saveto'] + '/adadelta_40k_out.txt'
    config['output_val_set'] = True
    config['beam_size'] = 20

    # Timing/monitoring related
    config['reload'] = True
    config['save_freq'] = 50
    config['sampling_freq'] = 1
    config['bleu_val_freq'] = 2000
    config['val_burn_in'] = 50000
    config['hook_samples'] = 1

    return config
