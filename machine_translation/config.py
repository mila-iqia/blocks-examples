def get_config_cs2en():
    config = {}

    # Model related -----------------------------------------------------------

    # Sequences longer than this will be discarded
    config['seq_len'] = 50

    # Number of hidden units in encoder/decoder GRU
    config['enc_nhids'] = 100
    config['dec_nhids'] = 100

    # Dimension of the word embedding matrix in encoder/decoder
    config['enc_embed'] = 62
    config['dec_embed'] = 62

    # Where to save model
    config['saveto'] = 'search_model_cs2en'

    # Optimization related ----------------------------------------------------

    # Batch size
    config['batch_size'] = 80

    # This many batches will be read ahead and sorted
    config['sort_k_batches'] = 12

    # Optimization step rule
    config['step_rule'] = 'AdaDelta'

    # Gradient clipping threshold
    config['step_clipping'] = 1

    # Std of weight initialization
    config['weight_scale'] = 0.01

    # Regularization related --------------------------------------------------

    # Weight noise flag for feed forward layers
    config['weight_noise_ff'] = False

    # Weight noise flag for recurrent layers
    config['weight_noise_rec'] = False

    # Dropout ratio, applied only after readout maxout
    config['dropout'] = 1.0

    # Vocabulary/dataset related ----------------------------------------------

    # Root directory for dataset
    datadir = '/data/lisatmp3/firatorh/nmt/wmt15/data/cs-en/'

    # Module name of the stream that will be used
    config['stream'] = 'stream_cs2en'

    # Source and target vocabularies
    config['src_vocab'] = datadir + 'all.tok.clean.shuf.cs-en.cs.vocab.pkl'
    config['trg_vocab'] = datadir + 'all.tok.clean.shuf.cs-en.en.vocab.pkl'

    # Source and target datasets
    config['src_data'] = datadir + 'all.tok.clean.shuf.cs-en.cs'
    config['trg_data'] = datadir + 'all.tok.clean.shuf.cs-en.en'

    # Source and target vocabulary sizes
    config['src_vocab_size'] = 40000
    config['trg_vocab_size'] = 40000

    # Special tokens and indexes
    config['unk_id'] = 1
    config['bos_token'] = '<S>'
    config['eos_token'] = '</S>'
    config['unk_token'] = '<UNK>'

    # Early stopping based on bleu related ------------------------------------

    # Normalize cost according to sequence length after beam-search
    config['normalized_bleu'] = True

    # Bleu script that will be used (moses multi-perl in this case)
    config['bleu_script'] = None #datadir + 'multi-bleu.perl'

    # Validation set source file
    config['val_set'] = datadir + 'newstest2013.tok.cs'

    # Validation set gold file
    config['val_set_grndtruth'] = datadir + 'newstest2013.tok.en'

    # Print validation output to file
    config['output_val_set'] = True

    # Validation output file
    config['val_set_out'] = config['saveto'] + '/adadelta_40k_out.txt'

    # Beam-size
    config['beam_size'] = 20

    # Timing/monitoring related -----------------------------------------------

    # Reload model from files if exist
    config['reload'] = False

    # Save model after this many updates
    config['save_freq'] = 50

    # Show samples from model after this many updates
    config['sampling_freq'] = 1

    # Show this many samples at each sampling
    config['hook_samples'] = 0

    # Validate bleu after this many updates
    config['bleu_val_freq'] = 2000

    # Start bleu validation after this many updates
    config['val_burn_in'] = 50000

    return config
