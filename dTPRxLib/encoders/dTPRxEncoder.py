"""
An LSTM with Recurrent Dropout and the option to use highway
connections between layers.
"""

from typing import Optional, Tuple
from overrides import overrides
import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.nn.util import get_dropout_mask
from dTPRxLib.cells.dTPRxCell import dTPRxCell
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import PytorchSeq2SeqWrapper
from dTPRxLib.cells.cell import Cell


@Seq2SeqEncoder.register("dTPRxEncoder")
class dTPRxEncoder(Seq2SeqEncoder):
    """
    An LSTM with Recurrent Dropout and the option to use highway
    connections between layers. Note: this implementation is slower
    than the native Pytorch LSTM because it cannot make use of CUDNN
    optimizations for stacked RNNs due to the highway layers and
    variational dropout.

    Parameters
    ----------
    input_size : int, required.
        The dimension of the inputs to the LSTM.
    hidden_size : int, required.
        The dimension of the outputs of the LSTM.
    go_forward: bool, optional (default = True)
        The direction in which the LSTM is applied to the sequence.
        Forwards by default, or backwards if False.
    recurrent_dropout_probability: float, optional (default = 0.0)
        The dropout probability to be used in a dropout scheme as stated in
        `A Theoretically Grounded Application of Dropout in Recurrent Neural Networks
        <https://arxiv.org/abs/1512.05287>`_ . Implementation wise, this simply
        applies a fixed dropout mask per sequence to the recurrent connection of the
        LSTM.
    use_highway: bool, optional (default = True)
        Whether or not to use highway connections between layers. This effectively involves
        reparameterising the normal output of an LSTM as::

            gate = sigmoid(W_x1 * x_t + W_h * h_t)
            output = gate * h_t  + (1 - gate) * (W_x2 * x_t)
    use_input_projection_bias : bool, optional (default = True)
        Whether or not to use a bias on the input projection layer. This is mainly here
        for backwards compatibility reasons and will be removed (and set to False)
        in future releases.

    Returns
    -------
    output_accumulator : PackedSequence
        The outputs of the LSTM for each timestep. A tensor of shape
        (batch_size, max_timesteps, hidden_size) where for a given batch
        element, all outputs past the sequence length for that batch are
        zero tensors.
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 cell: Cell,
                 go_forward: bool = True,
                 recurrent_dropout_probability: float = 0.0,
                 use_input_projection_bias: bool = True) -> None:
        super(dTPRxEncoder, self).__init__()
        # Required to be wrapped with a :class:`PytorchSeq2SeqWrapper`.
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = cell
        self.go_forward = go_forward
        self.recurrent_dropout_probability = recurrent_dropout_probability

    @overrides
    def get_input_dim(self) -> int:
        return self.input_size

    @overrides
    def get_output_dim(self) -> int:
        return self.hidden_size

    def forward(self,  # pylint: disable=arguments-differ
                inputs: PackedSequence,
                initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        Parameters
        ----------
        inputs : PackedSequence, required.
            A tensor of shape (batch_size, num_timesteps, input_size)
            to apply the LSTM over.

        initial_state : Tuple[torch.Tensor, torch.Tensor], optional, (default = None)
            A tuple (state, memory) representing the initial hidden state and memory
            of the LSTM. Each tensor has shape (1, batch_size, output_dimension).

        Returns
        -------
        A PackedSequence containing a torch.FloatTensor of shape
        (batch_size, num_timesteps, output_dimension) representing
        the outputs of the LSTM per timestep and a tuple containing
        the LSTM state, with shape (1, batch_size, hidden_size) to
        match the Pytorch API.
        """
        if not isinstance(inputs, PackedSequence):
            raise ConfigurationError('inputs must be PackedSequence but got %s' % (type(inputs)))

        sequence_tensor, batch_lengths = pad_packed_sequence(inputs, batch_first=True)
        batch_size = sequence_tensor.size()[0]
        total_timesteps = sequence_tensor.size()[1]

        # We have to use this '.data.new().resize_.fill_' pattern to create tensors with the correct
        # type - forward has no knowledge of whether these are torch.Tensors or torch.cuda.Tensors.
        output_accumulator = Variable(sequence_tensor.data.new()
                                      .resize_(batch_size, total_timesteps, self.hidden_size).fill_(0))
        if initial_state is None:
            full_batch_previous_state = Variable(sequence_tensor.data.new()
                                                 .resize_(batch_size, self.hidden_size).fill_(0))
        else:
            full_batch_previous_state = initial_state[0].squeeze(0)

        current_length_index = batch_size - 1 if self.go_forward else 0

        dropout_mask = None

        for timestep in range(total_timesteps):
            # The index depends on which end we start.
            index = timestep if self.go_forward else total_timesteps - timestep - 1

            # What we are doing here is finding the index into the batch dimension
            # which we need to use for this timestep, because the sequences have
            # variable length, so once the index is greater than the length of this
            # particular batch sequence, we no longer need to do the computation for
            # this sequence. The key thing to recognise here is that the batch inputs
            # must be _ordered_ by length from longest (first in batch) to shortest
            # (last) so initially, we are going forwards with every sequence and as we
            # pass the index at which the shortest elements of the batch finish,
            # we stop picking them up for the computation.
            if self.go_forward:
                while batch_lengths[current_length_index] <= index:
                    current_length_index -= 1
            # If we're going backwards, we are _picking up_ more indices.

            # Actually get the slices of the batch which we need for the computation at this timestep.
            previous_state = full_batch_previous_state[0: current_length_index + 1].clone()
            timestep_input = sequence_tensor[0: current_length_index + 1, index]

            timestep_output = self.cell(previous_state,timestep_input)

            # Only do dropout if the dropout prob is > 0.0 and we are in training mode.
            if dropout_mask is not None and self.training:
                timestep_output = timestep_output * dropout_mask[0: current_length_index + 1]

            # We've been doing computation with less than the full batch, so here we create a new
            # variable for the the whole batch at this timestep and insert the result for the
            # relevant elements of the batch into it.

            full_batch_previous_state = Variable(full_batch_previous_state.data.clone())
            full_batch_previous_state[0:current_length_index + 1] = timestep_output
            output_accumulator[0:current_length_index + 1, index] = timestep_output

        output_accumulator = pack_padded_sequence(output_accumulator, batch_lengths, batch_first=True)

        # Mimic the pytorch API by returning state in the following shape:
        # (num_layers * num_directions, batch_size, hidden_size). As this
        # LSTM cannot be stacked, the first dimension here is just 1.
        final_state = (full_batch_previous_state.unsqueeze(0))

        return output_accumulator, final_state

    @classmethod
    def from_params(cls, params: Params) -> 'PytorchSeq2SeqWrapper':
        input_size = params.pop("input_size")
        hidden_size = params.pop("hidden_size")
        cell_params = params.pop("cell")
        cell = Cell.from_params(cell_params)
        return PytorchSeq2SeqWrapper(cls(input_size=input_size,
                   hidden_size=hidden_size,cell=cell))