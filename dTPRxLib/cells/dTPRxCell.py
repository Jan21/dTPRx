
from typing import Dict

import numpy
from overrides import overrides

from allennlp.common import Params

from dTPRxLib.cells.cell import Cell


import torch

from allennlp.nn.initializers import block_orthogonal

@Cell.register('dTPRxCell')
class dTPRxCell(Cell):

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 iterations_per_cell: int = 3,
                 use_input_projection_bias: bool = True) -> None:
        super(dTPRxCell, self).__init__()
        # Required to be wrapped with a :class:`PytorchSeq2SeqWrapper`.

        self.iterations_per_cell = iterations_per_cell
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_for_r = torch.nn.Linear(hidden_size + input_size, hidden_size, bias=True) # linear for unbinder r
        self.linear_for_f = torch.nn.Linear(hidden_size + hidden_size, hidden_size, bias=True) # linear for filler f
        self.linear_for_new_f = torch.nn.Linear(hidden_size + hidden_size + input_size, hidden_size, bias=True) # linear for new filler f
        self.linear_for_new_r = torch.nn.Linear(hidden_size + input_size, hidden_size, bias=True) # linear for new role r
        self.linear_for_b = torch.nn.Linear(hidden_size + hidden_size, hidden_size, bias=True) # linear for new binding b
        self.nonlinearity = torch.nn.Tanh()
        self.reset_parameters()

    def reset_parameters(self):
        # Use sensible default initializations for parameters.
        block_orthogonal(self.linear_for_r.weight.data, [self.hidden_size,self.hidden_size+self.input_size])
        block_orthogonal(self.linear_for_f.weight.data, [self.hidden_size,self.hidden_size])
        block_orthogonal(self.linear_for_new_f.weight.data, [self.hidden_size, self.hidden_size+self.input_size+self.hidden_size])
        block_orthogonal(self.linear_for_new_r.weight.data, [self.hidden_size,self.hidden_size+self.input_size])
        block_orthogonal(self.linear_for_b.weight.data, [self.hidden_size, self.hidden_size])

    @overrides
    def get_output_dim(self) -> int:
        return self.hidden_size

    @overrides
    def forward(self,  # type: ignore
                hx,
                input_):
        # pylint: disable=arguments-differ


        for i in range(self.iterations_per_cell):
            input_hidden = torch.cat([input_, hx], 1) # first dimension is batch, second is size of the vector
            r_unbinder =  self.nonlinearity(self.linear_for_r(input_hidden))
            r_hidden = torch.cat([hx, r_unbinder ], 1)
            f_filler =  self.nonlinearity(self.linear_for_f(r_hidden))
            f_input_hidden = torch.cat([f_filler, input_, hx], 1)
            new_f_filler =  f_filler + self.nonlinearity(self.linear_for_new_f(f_input_hidden))
            new_f_filler = torch.clamp(new_f_filler, min=-1, max=1)
            new_r_unbinder = self.nonlinearity(self.linear_for_new_r(input_hidden))
            new_f_new_r = torch.cat([new_f_filler, new_r_unbinder], 1)
            b_binding = self.nonlinearity(self.linear_for_b(new_f_new_r))
            hx = hx+b_binding

        return hx

    @classmethod
    def from_params(cls, params: Params) -> 'dTPRxCell':
        input_size = params.pop("input_size")
        hidden_size = params.pop("hidden_size")
        return cls(input_size=input_size,
                   hidden_size=hidden_size)




