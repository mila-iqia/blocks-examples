Learn to reverse the letters of each individual word in a text
================================================================

In this demo, a recurrent network equipped with an attention mechanism
learns to reverse each word (on a character-by-character basis) in its input text. 

The default training data is the Google Billion Word corpus, 
which you should download and put to the path indicated in your .fuelrc file.

As an example, the first sentence should be transformed into the second:

* "The quick brown fox jumps over the lazy dog." <- INPUT
* "eht kciuq nworb xof spmuj revo eht yzal god." <- OUTPUT


The bulk of the functionality of the code is in the ``__init__.py`` file.


Structure of the Data
--------------------------

The input data arrives through the a ``fuel`` data processing pipeline,
and the first section of the code deals with functions that clean and prepare the
data :

* converting character codes to and from a numerical (integer) encoding 
* a 'gold standard' reverse_words function that performs the task perfectly
* ``_lower``, ``_filter`` and ``_is_nan`` data-cleaning functions
* ``_transpose`` function that *TO-FIGURE-OUT*

Once the data is read in character-wise (``level="character"``), it
is cleaned up and converted into mini-batches using a ``ConstantScheme(10)``, which 
*TO-FIGURE-OUT*


Structure of the Model
--------------------------

``class WordReverser`` is initialized with :

* ``dimension`` which refers to the size of the hidden size of the internal state-to-state data
* ``alphabet_size`` which allows the initial vector embedding (via ``LookupTable``) of characters into 
  a ``dimension`` dimensional vector (so that ``dimension`` is doing double-duty as the
  size of the internal recurrence, and the size of the vector embedding)

The model itself is ``Bidirectional``, with ``SimpleRecurrent`` units.  This means
*TO-FIGURE-OUT*

At each time-step :

* the current character is mapped to an embedding vector
* the input has ``Fork`` applied, so that *TO-FIGURE-OUT*
* and there is an attention mechanism (using ``SequenceContentAttention``) that 
  connects to *TO-FIGURE-OUT*
* the final output is read via a ``SoftmaxEmitter`` and converted to characters 
* and is then fed back to the input via a `LookupFeedback` *TO-FIGURE-OUT*


Structure of the Training
--------------------------

The training algorithm uses ``GradientDescent`` with a maximum 
step-size (``StepClipping(10.0)``) applied to 
the mean over 
the batch of 
the sums of 
the log-likelihood costs associated with 
errors on the character outputs for 
each sentence.


Structure of the Testing
--------------------------

There are two modes available, which are defined within the ``generate(_input)`` 
local function definition: 

* ``beam_search`` which does a ``BeamSearch`` for the most likely output sequence, given an input sentence
* ``sample`` which instead produces a number of guesses at the output sentence

