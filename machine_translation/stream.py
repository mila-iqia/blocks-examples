# Currently works with https://github.com/orhanf/picklable_itertools
# TODO: Remove parametrized ifilter dependency
# TODO: Remove MappingWithArgs
# TODO: Remove FilterWithArgs
import cPickle

from fuel.datasets import TextFile
from fuel.schemes import ConstantScheme
from fuel.streams import DataStream
from fuel.transformers import (
    Merge, Batch, Filter, Padding, SortMapping, Unpack, Mapping, Transformer)
from picklable_itertools import ifilter


# Everthing here should be wrapped and parameterized by config
# this import is to workaround for pickling errors when wrapped
# TODO: find a better solution
from model import config


class MappingWithArgs(Mapping):
    """Extend Mapping in order to pass arguments to mapping function."""
    def __init__(self, data_stream, mapping, add_sources=None,
                 **kwargs):
        super(MappingWithArgs, self).__init__(
            data_stream, mapping, add_sources)
        self.mapping_args = kwargs

    def get_data(self, request=None):
        if request is not None:
            raise ValueError
        data = next(self.child_epoch_iterator)
        image = self.mapping(data, **self.mapping_args)
        if not self.add_sources:
            return image
        return data + image


class FilterWithArgs(Transformer):
    """Filters samples that meet a predicate."""
    def __init__(self, data_stream, predicate, predicate_args=None):
        super(FilterWithArgs, self).__init__(data_stream)
        self.predicate = predicate
        self.predicate_args = predicate_args if predicate_args else {}

    def get_epoch_iterator(self, **kwargs):
        super(FilterWithArgs, self).get_epoch_iterator(**kwargs)
        return ifilter(self.predicate, self.child_epoch_iterator,
                       **self.predicate_args)


def _length(sentence_pair):
    return len(sentence_pair[1])


def _oov_to_unk(sentence_pair, src_vocab_size=30000,
                trg_vocab_size=30000, unk_id=1):
    src_vocab_size = src_vocab_size
    trg_vocab_size = trg_vocab_size
    unk_id = unk_id
    return ([x if x < src_vocab_size else unk_id for x in sentence_pair[0]],
            [x if x < trg_vocab_size else unk_id for x in sentence_pair[1]])


def _too_long(sentence_pair, seq_len=50):
    return all([len(sentence) <= seq_len
                for sentence in sentence_pair])

cs_vocab = config['src_vocab']
en_vocab = config['trg_vocab']
cs_file = config['src_data']
en_file = config['trg_data']

cs_dataset = TextFile([cs_file], cPickle.load(open(cs_vocab)), None)
en_dataset = TextFile([en_file], cPickle.load(open(en_vocab)), None)

stream = Merge([cs_dataset.get_example_stream(),
                en_dataset.get_example_stream()],
               ('source', 'target'))

stream = Filter(stream, predicate=_too_long,
                predicate_args={'seq_len': config['seq_len']})
stream = MappingWithArgs(
    stream, _oov_to_unk, src_vocab_size=config['src_vocab_size'],
    trg_vocab_size=config['trg_vocab_size'], unk_id=config['unk_id'])
stream = Batch(stream,
               iteration_scheme=ConstantScheme(
                   config['batch_size']*config['sort_k_batches']))

stream = Mapping(stream, SortMapping(_length))
stream = Unpack(stream)
stream = Batch(stream, iteration_scheme=ConstantScheme(config['batch_size']))
masked_stream = Padding(stream)

# Setup development set stream if necessary
dev_stream = None
if 'val_set' in config and config['val_set']:
    dev_file = config['val_set']
    dev_dataset = TextFile([dev_file], cPickle.load(open(cs_vocab)), None)
    dev_stream = DataStream(dev_dataset)
