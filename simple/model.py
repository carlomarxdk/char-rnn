import torch
import torch.nn as nn


class CharRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(CharRNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.encoder = nn.Embedding(num_embeddings=self.output_size,
                                    embedding_dim=self.hidden_size)
        self.rnn = nn.GRU(input_size=self.hidden_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True,
                          dropout=0.7)

        self.decoder = nn.Sequential(nn.Linear(in_features=self.hidden_size,
                                               out_features=self.output_size),
                                     nn.Softmax(dim=1))

    def forward(self, x, hidden):
        #batch_size = x.shape[0]
        x = self.encoder(x)
        output, hidden = self.rnn(x, hidden)
        output = self.decoder(output)
        return output, hidden

    def predict(self, x, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(1)
        output, hidden = self.forward(x, hidden)
        return output.topk(1)[1], hidden

    def init_hidden(self, batch_size):
        return torch.zeros([self.num_layers, batch_size, self.hidden_size])

    def sample(self, in_sequence):
        hidden = self.init_hidden(1)

        out_sequence = list()
        output = 0
        for char in in_sequence:
            output, hidden = self.predict(char, hidden)
            out_sequence.append(char.data.numpy())

        # sample the sequence
        for ii in range(32):
            output, hidden = self.predict(output.unsqueeze(0), hidden)
            out_sequence.append(output.data.numpy())

        return out_sequence

