"""
Modules that transform a sequence of input vectors
into a single output vector.
Some are just basic wrappers around existing PyTorch modules,
others are AllenNLP modules.

The available Seq2Vec encoders are

* `"gru" <http://pytorch.org/docs/master/nn.html#torch.nn.GRU>`_
* `"lstm" <http://pytorch.org/docs/master/nn.html#torch.nn.LSTM>`_
* `"rnn" <http://pytorch.org/docs/master/nn.html#torch.nn.RNN>`_
* :class:`"cnn" <allennlp.modules.seq2vec_encoders.cnn_encoder.CnnEncoder>`
* :class:`"augmented_lstm" <allennlp.modules.augmented_lstm.AugmentedLstm>`
* :class:`"alternating_lstm" <allennlp.modules.stacked_alternating_lstm.StackedAlternatingLstm>`
"""


#from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.modules.seq2seq_encoders import _Seq2SeqWrapper
from dTPRxLib.encoders.dTPRxEncoder import dTPRxEncoder


# pylint: disable=protected-access
#Seq2SeqEncoder.register("dTPRxEncoder")(_Seq2SeqWrapper(dTPRxEncoder))
