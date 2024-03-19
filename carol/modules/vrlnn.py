'''
Wrap up the neural network modules.
'''

import math

import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.parameter import Parameter
from typing import List, Sequence, Tuple


class Linear(nn.Linear):
    def forward(self, input):
        res = input.matmul(self.weight.t()).add(self.bias)
        return res

class EnsembleLinearLayer(nn.Module):
    def __init__(
        self, num_members: int, in_size: int, out_size: int, bias: bool = True
    ):
        super().__init__()
        self.num_members = num_members
        self.in_size = in_size
        self.out_size = out_size
        self.weight = nn.Parameter(
            torch.rand(self.num_members, self.in_size, self.out_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.rand(self.num_members, 1, self.out_size))
            self.use_bias = True
        else:
            self.use_bias = False

        self.elite_models: List[int] = None

    def forward(self, x):
        xw = x.matmul(self.weight)
        if self.use_bias:
            return xw.add(self.bias)
        else:
            return xw

    def extra_repr(self) -> str:
        return (
            f"num_members={self.num_members}, in_size={self.in_size}, "
            f"out_size={self.out_size}, bias={self.use_bias}"
        )

    def set_elite(self, elite_models: Sequence[int]):
        self.elite_models = list(elite_models)

    def toggle_use_only_elite(self):
        self.use_only_elite = not self.use_only_elite



class ReLU(nn.Module):
    def __init__(self, inplace: bool=False):
        super(ReLU, self).__init__()
        self.inplace = inplace
    
    def forward(self, input):
        return input.relu()


class Tanh(nn.Module):
    def forward(self, input):
        return input.tanh()


