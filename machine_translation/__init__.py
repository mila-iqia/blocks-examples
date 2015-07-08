import logging
import numpy
import os
import time

from collections import Counter
from contextlib import closing
from theano import tensor
from toolz import merge

from blocks.algorithms import (GradientDescent, StepClipping, AdaDelta,
                               CompositeRule)
from blocks.bricks import (Tanh, Maxout, Linear, FeedforwardSequence,
                           Bias, Initializable, MLP)
from blocks.bricks.attention import SequenceContentAttention
from blocks.bricks.base import application
from blocks.bricks.lookup import LookupTable
from blocks.bricks.parallel import Fork
from blocks.bricks.recurrent import GatedRecurrent, Bidirectional
from blocks.bricks.sequence_generators import (
    LookupFeedback, Readout, SoftmaxEmitter,
    SequenceGenerator)
from blocks.extensions.saveload import SAVED_TO, LOADED_FROM
from blocks.extensions import (
    FinishAfter, Printing, TrainingExtension, SimpleExtension)
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.extensions.plot import Plot
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_noise, apply_dropout
from blocks.initialization import IsotropicGaussian, Orthogonal, Constant
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.roles import add_role, WEIGHT
from blocks.select import Selector
from blocks.serialization import secure_dump, load, BRICK_DELIMITER
from blocks.utils import shared_floatx_nans, reraise_as

from picklable_itertools.extras import equizip
from sampling import BleuValidator, Sampler


logger = logging.getLogger(__name__)


class SaveLoadUtils(object):
    """Utility class for checkpointing."""

    @property
    def path_to_parameters(self):
        return os.path.join(self.folder, 'params.npz')

    @property
    def path_to_iteration_state(self):
        return os.path.join(self.folder, 'iterations_state.pkl')

    @property
    def path_to_log(self):
        return os.path.join(self.folder, 'log')

    def load_parameter_values(self, path):
        with closing(numpy.load(path)) as source:
            param_values = {}
            for name, value in source.items():
                if name != 'pkl':
                    name_ = name.replace(BRICK_DELIMITER, '/')
                    if not name_.startswith('/'):
                        name_ = '/' + name_
                    param_values[name_] = value
        return param_values

    def save_parameter_values(self, param_values, path):
        param_values = {name.replace("/", "-"): param
                        for name, param in param_values.items()}
        numpy.savez(path, **param_values)


class CheckpointNMT(SimpleExtension, SaveLoadUtils):
    """Redefines checkpointing for NMT.

        Saves only parameters (npz), iteration state (pickle) and log (pickle).

    """

    def __init__(self, saveto, **kwargs):
        self.folder = saveto
        kwargs.setdefault("after_training", True)
        super(CheckpointNMT, self).__init__(**kwargs)

    def dump_parameters(self, main_loop):
        params_to_save = main_loop.model.get_parameter_values()
        self.save_parameter_values(params_to_save,
                                   self.path_to_parameters)

    def dump_iteration_state(self, main_loop):
        secure_dump(main_loop.iteration_state, self.path_to_iteration_state)

    def dump_log(self, main_loop):
        secure_dump(main_loop.log, self.path_to_log)

    def dump(self, main_loop):
        if not os.path.exists(self.folder):
            os.mkdir(self.folder)
        print ""
        logger.info(" Saving model")
        start = time.time()
        logger.info(" ...saving parameters")
        self.dump_parameters(main_loop)
        logger.info(" ...saving iteration state")
        self.dump_iteration_state(main_loop)
        logger.info(" ...saving log")
        self.dump_log(main_loop)
        logger.info(" Model saved, took {} seconds.".format(time.time()-start))

    def do(self, callback_name, *args):
        try:
            self.dump(self.main_loop)
        except Exception:
            raise
        finally:
            already_saved_to = self.main_loop.log.current_row.get(SAVED_TO, ())
            self.main_loop.log.current_row[SAVED_TO] = (already_saved_to +
                                                        (self.folder,))


class LoadNMT(TrainingExtension, SaveLoadUtils):
    """Loads parameters log and iterations state."""

    def __init__(self, saveto, **kwargs):
        self.folder = saveto
        super(LoadNMT, self).__init__(saveto, **kwargs)

    def before_training(self):
        if not os.path.exists(self.folder):
            logger.info("No dump found")
            return
        logger.info("Loading the state from {} into the main loop"
                    .format(self.folder))
        try:
            self.load_to(self.main_loop)
            self.main_loop.log.current_row[LOADED_FROM] = self.folder
        except Exception:
            reraise_as("Failed to load the state")

    def load_parameters(self):
        return self.load_parameter_values(self.path_to_parameters)

    def load_iteration_state(self):
        with open(self.path_to_iteration_state, "rb") as source:
            return load(source)

    def load_log(self):
        with open(self.path_to_log, "rb") as source:
            return load(source)

    def load(self):
        return (self.load_parameters(),
                self.load_iteration_state(),
                self.load_log())

    def load_to(self, main_loop):
        """Loads the dump from the root folder into the main loop."""
        logger.info(" Reloading model")
        try:
            logger.info(" ...loading model parameters")
            params_all = self.load_parameters()
            params_this = main_loop.model.get_parameter_dict()
            missing = set(params_this.keys()) - set(params_all.keys())
            for pname in params_this.keys():
                if pname in params_all:
                    val = params_all[pname]
                    if params_this[pname].get_value().shape != val.shape:
                        logger.warning(
                            " Dimension mismatch {}-{} for {}"
                            .format(params_this[pname].get_value().shape,
                                    val.shape, pname))

                    params_this[pname].set_value(val)
                    logger.info(" Loaded to CG {:15}: {}"
                                .format(val.shape, pname))
                else:
                    logger.warning(
                        " Parameter does not exist: {}".format(pname))
            logger.info(
                " Number of parameters loaded for computation graph: {}"
                .format(len(params_this) - len(missing)))
        except Exception as e:
            logger.error(" Error {0}".format(str(e)))

        try:
            logger.info(" Loading iteration state...")
            main_loop.iteration_state = self.load_iteration_state()
        except Exception as e:
            logger.error(" Error {0}".format(str(e)))

        try:
            logger.info(" Loading log...")
            main_loop.log = self.load_log()
        except Exception as e:
            logger.error(" Error {0}".format(str(e)))


# Helper class
class InitializableFeedforwardSequence(FeedforwardSequence, Initializable):
    pass


class LookupFeedbackWMT15(LookupFeedback):
    """Zero-out initial readout feedback by checking its value."""

    @application
    def feedback(self, outputs):
        assert self.output_dim == 0

        shp = [outputs.shape[i] for i in xrange(outputs.ndim)]
        outputs_flat = outputs.flatten()
        outputs_flat_zeros = tensor.switch(outputs_flat < 0, 0,
                                           outputs_flat)

        lookup_flat = tensor.switch(
            outputs_flat[:, None] < 0,
            tensor.alloc(0., outputs_flat.shape[0], self.feedback_dim),
            self.lookup.apply(outputs_flat_zeros))
        lookup = lookup_flat.reshape(shp+[self.feedback_dim])
        return lookup


class BidirectionalWMT15(Bidirectional):
    """Wrap two Gated Recurrents each having separate parameters."""

    @application
    def apply(self, forward_dict, backward_dict):
        """Applies forward and backward networks and concatenates outputs."""
        forward = self.children[0].apply(as_list=True, **forward_dict)
        backward = [x[::-1] for x in
                    self.children[1].apply(reverse=True, as_list=True,
                                           **backward_dict)]
        return [tensor.concatenate([f, b], axis=2)
                for f, b in equizip(forward, backward)]


class BidirectionalEncoder(Initializable):
    """Encoder of RNNsearch model."""

    def __init__(self, vocab_size, embedding_dim, state_dim, **kwargs):
        super(BidirectionalEncoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim

        self.lookup = LookupTable(name='embeddings')
        self.bidir = BidirectionalWMT15(
            GatedRecurrent(activation=Tanh(), dim=state_dim))
        self.fwd_fork = Fork(
            [name for name in self.bidir.prototype.apply.sequences
             if name != 'mask'], prototype=Linear(), name='fwd_fork')
        self.back_fork = Fork(
            [name for name in self.bidir.prototype.apply.sequences
             if name != 'mask'], prototype=Linear(), name='back_fork')

        self.children = [self.lookup, self.bidir,
                         self.fwd_fork, self.back_fork]

    def _push_allocation_config(self):
        self.lookup.length = self.vocab_size
        self.lookup.dim = self.embedding_dim

        self.fwd_fork.input_dim = self.embedding_dim
        self.fwd_fork.output_dims = [self.bidir.children[0].get_dim(name)
                                     for name in self.fwd_fork.output_names]
        self.back_fork.input_dim = self.embedding_dim
        self.back_fork.output_dims = [self.bidir.children[1].get_dim(name)
                                      for name in self.back_fork.output_names]

    @application(inputs=['source_sentence', 'source_sentence_mask'],
                 outputs=['representation'])
    def apply(self, source_sentence, source_sentence_mask):
        # Time as first dimension
        source_sentence = source_sentence.T
        source_sentence_mask = source_sentence_mask.T

        embeddings = self.lookup.apply(source_sentence)

        representation = self.bidir.apply(
            merge(self.fwd_fork.apply(embeddings, as_dict=True),
                  {'mask': source_sentence_mask}),
            merge(self.back_fork.apply(embeddings, as_dict=True),
                  {'mask': source_sentence_mask})
        )
        return representation


class GRUInitialState(GatedRecurrent):
    """Gated Recurrent with special initial state.

    Initial state of Gated Recurrent is set by an MLP that conditions on the
    last hidden state of the bidirectional encoder, applies an affine
    transformation followed by a tanh non-linearity to set initial state.

    """
    def __init__(self, attended_dim, **kwargs):
        super(GRUInitialState, self).__init__(**kwargs)
        self.attended_dim = attended_dim
        self.initial_transformer = MLP(activations=[Tanh()],
                                       dims=[attended_dim, self.dim],
                                       name='state_initializer')
        self.children.append(self.initial_transformer)

    @application
    def initial_states(self, batch_size, *args, **kwargs):
        attended = kwargs['attended']
        initial_state = self.initial_transformer.apply(
            attended[0, :, -self.attended_dim:])
        return initial_state

    def _allocate(self):
        self.parameters.append(shared_floatx_nans((self.dim, self.dim),
                               name='state_to_state'))
        self.parameters.append(shared_floatx_nans((self.dim, 2 * self.dim),
                               name='state_to_gates'))
        for i in range(2):
            if self.parameters[i]:
                add_role(self.parameters[i], WEIGHT)


class Decoder(Initializable):
    """Decoder of RNNsearch model."""

    def __init__(self, vocab_size, embedding_dim, state_dim,
                 representation_dim, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.representation_dim = representation_dim

        # Initialize gru with special initial state
        self.transition = GRUInitialState(
            attended_dim=state_dim, dim=state_dim,
            activation=Tanh(), name='decoder')

        # Initialize the attention mechanism
        self.attention = SequenceContentAttention(
            state_names=self.transition.apply.states,
            attended_dim=representation_dim,
            match_dim=state_dim, name="attention")

        # Initialize the readout, note that SoftmaxEmitter emits -1 for
        # initial outputs which is used by LookupFeedBackWMT15
        readout = Readout(
            source_names=['states', 'feedback',
                          self.attention.take_glimpses.outputs[0]],
            readout_dim=self.vocab_size,
            emitter=SoftmaxEmitter(initial_output=-1),
            feedback_brick=LookupFeedbackWMT15(vocab_size, embedding_dim),
            post_merge=InitializableFeedforwardSequence(
                [Bias(dim=state_dim, name='maxout_bias').apply,
                 Maxout(num_pieces=2, name='maxout').apply,
                 Linear(input_dim=state_dim / 2, output_dim=embedding_dim,
                        use_bias=False, name='softmax0').apply,
                 Linear(input_dim=embedding_dim, name='softmax1').apply]),
            merged_dim=state_dim)

        # Build sequence generator accordingly
        self.sequence_generator = SequenceGenerator(
            readout=readout,
            transition=self.transition,
            attention=self.attention,
            fork=Fork([name for name in self.transition.apply.sequences
                       if name != 'mask'], prototype=Linear())
        )

        self.children = [self.sequence_generator]

    @application(inputs=['representation', 'source_sentence_mask',
                         'target_sentence_mask', 'target_sentence'],
                 outputs=['cost'])
    def cost(self, representation, source_sentence_mask,
             target_sentence, target_sentence_mask):

        source_sentence_mask = source_sentence_mask.T
        target_sentence = target_sentence.T
        target_sentence_mask = target_sentence_mask.T

        # Get the cost matrix
        cost = self.sequence_generator.cost_matrix(**{
            'mask': target_sentence_mask,
            'outputs': target_sentence,
            'attended': representation,
            'attended_mask': source_sentence_mask}
        )

        return (cost * target_sentence_mask).sum() / \
            target_sentence_mask.shape[1]

    @application
    def generate(self, source_sentence, representation):
        return self.sequence_generator.generate(
            n_steps=2 * source_sentence.shape[1],
            batch_size=source_sentence.shape[0],
            attended=representation,
            attended_mask=tensor.ones(source_sentence.shape).T,
            glimpses=self.attention.take_glimpses.outputs[0])


def main(config, tr_stream, dev_stream, bokeh=False):

    # Create Theano variables
    logger.info('Creating theano variables')
    source_sentence = tensor.lmatrix('source')
    source_sentence_mask = tensor.matrix('source_mask')
    target_sentence = tensor.lmatrix('target')
    target_sentence_mask = tensor.matrix('target_mask')
    sampling_input = tensor.lmatrix('input')

    # Construct model
    logger.info('Building RNN encoder-decoder')
    encoder = BidirectionalEncoder(
        config['src_vocab_size'], config['enc_embed'], config['enc_nhids'])
    decoder = Decoder(
        config['trg_vocab_size'], config['dec_embed'], config['dec_nhids'],
        config['enc_nhids'] * 2)
    cost = decoder.cost(
        encoder.apply(source_sentence, source_sentence_mask),
        source_sentence_mask, target_sentence, target_sentence_mask)

    logger.info('Creating computational graph')
    cg = ComputationGraph(cost)

    # Initialize model
    logger.info('Initializing model')
    encoder.weights_init = decoder.weights_init = IsotropicGaussian(
        config['weight_scale'])
    encoder.biases_init = decoder.biases_init = Constant(0)
    encoder.push_initialization_config()
    decoder.push_initialization_config()
    encoder.bidir.prototype.weights_init = Orthogonal()
    decoder.transition.weights_init = Orthogonal()
    encoder.initialize()
    decoder.initialize()

    # apply dropout for regularization
    if config['dropout'] < 1.0:
        # dropout is applied to the output of maxout in ghog
        logger.info('Applying dropout')
        dropout_inputs = [x for x in cg.intermediary_variables
                          if x.name == 'maxout_apply_output']
        cg = apply_dropout(cg, dropout_inputs, config['dropout'])

    # Apply weight noise for regularization
    if config['weight_noise_ff'] > 0.0:
        logger.info('Applying weight noise to ff layers')
        enc_params = Selector(encoder.lookup).get_params().values()
        enc_params += Selector(encoder.fwd_fork).get_params().values()
        enc_params += Selector(encoder.back_fork).get_params().values()
        dec_params = Selector(
            decoder.sequence_generator.readout).get_params().values()
        dec_params += Selector(
            decoder.sequence_generator.fork).get_params().values()
        dec_params += Selector(decoder.state_init).get_params().values()
        cg = apply_noise(cg, enc_params+dec_params, config['weight_noise_ff'])

    # Print shapes
    shapes = [param.get_value().shape for param in cg.parameters]
    logger.info("Parameter shapes: ")
    for shape, count in Counter(shapes).most_common():
        logger.info('    {:15}: {}'.format(shape, count))
    logger.info("Total number of parameters: {}".format(len(shapes)))

    # Print parameter names
    enc_dec_param_dict = merge(Selector(encoder).get_parameters(),
                               Selector(decoder).get_parameters())
    logger.info("Parameter names: ")
    for name, value in enc_dec_param_dict.iteritems():
        logger.info('    {:15}: {}'.format(value.get_value().shape, name))
    logger.info("Total number of parameters: {}"
                .format(len(enc_dec_param_dict)))

    # Set up training model
    logger.info("Building model")
    training_model = Model(cost)

    # Set extensions
    logger.info("Initializing extensions")
    extensions = [
        FinishAfter(after_n_batches=config['finish_after']),
        TrainingDataMonitoring([cost], after_batch=True),
        Printing(after_batch=True),
        CheckpointNMT(config['saveto'],
                      every_n_batches=config['save_freq'])
    ]

    # Set up beam search and sampling computation graphs if necessary
    if config['hook_samples'] > 1 or config['bleu_script'] is not None:
        logger.info("Building sampling model")
        sampling_representation = encoder.apply(
            sampling_input, tensor.ones(sampling_input.shape))
        generated = decoder.generate(sampling_input, sampling_representation)
        search_model = Model(generated)
        _, samples = VariableFilter(
            bricks=[decoder.sequence_generator], name="outputs")(
                ComputationGraph(generated[1]))  # generated[1] is next_outputs

    # Add sampling
    if config['hook_samples'] > 1:
        logger.info("Building sampler")
        extensions.append(
            Sampler(model=search_model, config=config, data_stream=tr_stream,
                    every_n_batches=config['sampling_freq']))

    # Add early stopping based on bleu
    if config['bleu_script'] is not None:
        logger.info("Building bleu validator")
        BleuValidator(sampling_input, samples=samples, config=config,
                      model=search_model, data_stream=dev_stream,
                      every_n_batches=config['bleu_val_freq'])

    # Reload model if necessary
    if config['reload']:
        extensions.append(LoadNMT(config['saveto']))

    # Plot cost in bokeh if necessary
    if bokeh:
        extensions.append(
            Plot('Cs-En', channels=[['decoder_cost_cost']],
                 after_batch=True))

    # Set up training algorithm
    logger.info("Initializing training algorithm")
    algorithm = GradientDescent(
        cost=cost, parameters=cg.parameters,
        step_rule=CompositeRule([StepClipping(config['step_clipping']),
                                 eval(config['step_rule'])()])
    )

    # Initialize main loop
    logger.info("Initializing main loop")
    main_loop = MainLoop(
        model=training_model,
        algorithm=algorithm,
        data_stream=tr_stream,
        extensions=extensions
    )

    # Train!
    main_loop.run()
