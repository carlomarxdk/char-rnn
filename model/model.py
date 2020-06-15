import torch
import torch.nn as nn


class CharRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, vocab):
        super(CharRNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab = vocab

        self.encoder = nn.Embedding(num_embeddings=self.vocab,
                                    embedding_dim=self.hidden_size)
        self.rnn = nn.GRU(input_size=self.hidden_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True,
                          dropout=0.7)

        self.decoder = nn.Sequential(nn.Linear(in_features=self.hidden_size,
                                               out_features=self.output_size),
                                     nn.LogSoftmax(dim=1))

    def forward(self, x, hidden):
        x = self.encoder(x)
        output, hidden = self.rnn(x, hidden)
        output = self.decoder(output)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.Variable(torch.zeros(batch_size, self.n_layers, self.hidden_size))
