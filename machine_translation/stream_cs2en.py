import cPickle

from fuel.datasets import TextFile
from fuel.schemes import ConstantScheme
from fuel.streams import DataStream
from fuel.transformers import (
    Merge, Batch, Filter, Padding, SortMapping, Unpack, Mapping)

from __main__ import config


def _length(sentence_pair):
    """Assumes target is the last element in the tuple."""
    return len(sentence_pair[-1])


class _oov_to_unk(object):
    """Maps out of vocabulary token index to unk token index."""
    def __init__(self, src_vocab_size=30000, trg_vocab_size=30000,
                 unk_id=1):
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.unk_id = unk_id

    def __call__(self, sentence_pair):
        return ([x if x < self.src_vocab_size else self.unk_id
                 for x in sentence_pair[0]],
                [x if x < self.trg_vocab_size else self.unk_id
                 for x in sentence_pair[1]])


class _too_long(object):
    """Filters sequences longer than given sequence length."""
    def __init__(self, seq_len=50):
        self.seq_len = seq_len

    def __call__(self, sentence_pair):
        return all([len(sentence) <= self.seq_len
                    for sentence in sentence_pair])

# Get helpers
cs_vocab_file = config['src_vocab']
en_vocab_file = config['trg_vocab']
cs_file = config['src_data']
en_file = config['trg_data']

# Load dictionaries and ensure special tokens exist
cs_vocab = cPickle.load(open(cs_vocab_file))
en_vocab = cPickle.load(open(en_vocab_file))
cs_vocab[config['bos_token']] = 0
cs_vocab[config['eos_token']] = config['src_vocab_size']
cs_vocab[config['unk_token']] = config['unk_id']
en_vocab[config['bos_token']] = 0
en_vocab[config['eos_token']] = config['trg_vocab_size']
en_vocab[config['unk_token']] = config['unk_id']

# Get text files from both source and target
cs_dataset = TextFile([cs_file], cs_vocab, None)
en_dataset = TextFile([en_file], en_vocab, None)

# Merge them to get a source, target pair
stream = Merge([cs_dataset.get_example_stream(),
                en_dataset.get_example_stream()],
               ('source', 'target'))

# Filter sequences that are too long
stream = Filter(stream,
                predicate=_too_long(seq_len=config['seq_len']))

# Replace out of vocabulary tokens with unk token
stream = Mapping(stream,
                 _oov_to_unk(src_vocab_size=config['src_vocab_size'],
                             trg_vocab_size=config['trg_vocab_size'],
                             unk_id=config['unk_id']))

# Build a batched version of stream to read k batches ahead
stream = Batch(stream,
               iteration_scheme=ConstantScheme(
                   config['batch_size']*config['sort_k_batches']))

# Sort all samples in the read-ahead batch
stream = Mapping(stream, SortMapping(_length))

# Convert it into a stream again
stream = Unpack(stream)

# Construct batches from the stream with specified batch size
stream = Batch(stream, iteration_scheme=ConstantScheme(config['batch_size']))

# Pad sequences that are short
masked_stream = Padding(stream)

# Setup development set stream if necessary
dev_stream = None
if 'val_set' in config and config['val_set']:
    dev_file = config['val_set']
    dev_dataset = TextFile([dev_file], cs_vocab, None)
    dev_stream = DataStream(dev_dataset)
