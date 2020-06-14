import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(CharRNN, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.rnn = nn.GRU(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers)

    def forward(self):
        raise NotImplementedError
