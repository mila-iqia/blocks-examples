import numpy
import tempfile
import theano

from blocks.initialization import IsotropicGaussian, Orthogonal, Constant
from blocks.model import Model

from machine_translation.model import BidirectionalEncoder, Decoder
from machine_translation.stream import get_tr_stream, get_dev_stream

from numpy.testing import assert_allclose


def test_search_model():

    # Create Theano variables
    floatX = theano.config.floatX
    source_sentence = theano.tensor.lmatrix('source')
    source_sentence_mask = theano.tensor.matrix('source_mask', dtype=floatX)
    target_sentence = theano.tensor.lmatrix('target')
    target_sentence_mask = theano.tensor.matrix('target_mask', dtype=floatX)

    # Construct model
    encoder = BidirectionalEncoder(
        vocab_size=10, embedding_dim=5, state_dim=8)
    decoder = Decoder(
        vocab_size=12, embedding_dim=6, state_dim=8, representation_dim=16)
    cost = decoder.cost(
        encoder.apply(source_sentence, source_sentence_mask),
        source_sentence_mask, target_sentence, target_sentence_mask)

    # Compile a function for the cost
    f_cost = theano.function(
        inputs=[source_sentence, source_sentence_mask,
                target_sentence, target_sentence_mask],
        outputs=cost)

    # Create literal variables
    numpy.random.seed(1234)
    x = numpy.random.randint(0, 10, size=(22, 4))
    y = numpy.random.randint(0, 12, size=(22, 6))
    x_mask = numpy.ones_like(x).astype(floatX)
    y_mask = numpy.ones_like(y).astype(floatX)

    # Initialize model
    encoder.weights_init = decoder.weights_init = IsotropicGaussian(
        0.01)
    encoder.biases_init = decoder.biases_init = Constant(0)
    encoder.push_initialization_config()
    decoder.push_initialization_config()
    encoder.bidir.prototype.weights_init = Orthogonal()
    decoder.transition.weights_init = Orthogonal()
    encoder.initialize()
    decoder.initialize()

    cost_ = f_cost(x, x_mask, y, y_mask)
    assert_allclose(cost_, 14.90944)


def test_stream():

    # Dummy vocabulary
    vocab = {'<S>': 0, '</S>': 1, '<UNK>': 2}
    with tempfile.NamedTemporaryFile() as src_data:
        with tempfile.NamedTemporaryFile() as trg_data:
            get_tr_stream(
                src_vocab=vocab, trg_vocab=vocab, src_data=src_data.name,
                trg_data=trg_data.name)
    with tempfile.NamedTemporaryFile() as val_set:
        get_dev_stream(val_set=val_set.name, src_vocab=vocab)


def test_sampling():

    # Create Theano variables
    sampling_input = theano.tensor.lmatrix('input')

    # Construct model
    encoder = BidirectionalEncoder(
        vocab_size=10, embedding_dim=5, state_dim=8)
    decoder = Decoder(
        vocab_size=12, embedding_dim=6, state_dim=8, representation_dim=16,
        theano_seed=1234)
    sampling_representation = encoder.apply(
        sampling_input, theano.tensor.ones(sampling_input.shape))
    generateds = decoder.generate(sampling_input, sampling_representation)
    model = Model(generateds[1])

    # Initialize model
    encoder.weights_init = decoder.weights_init = IsotropicGaussian(
        0.01)
    encoder.biases_init = decoder.biases_init = Constant(0)
    encoder.push_initialization_config()
    decoder.push_initialization_config()
    encoder.bidir.prototype.weights_init = Orthogonal()
    decoder.transition.weights_init = Orthogonal()
    encoder.initialize()
    decoder.initialize()

    # Compile a function for the generated
    sampling_fn = model.get_theano_function()

    # Create literal variables
    numpy.random.seed(1234)
    x = numpy.random.randint(0, 10, size=(1, 2))

    # Call function and check result
    generated_step = sampling_fn(x)
    assert len(generated_step[0].flatten()) == 4
