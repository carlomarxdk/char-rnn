import torch
import numpy as np
from urllib import request
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    # Dostoyevsky: Brothers Karamazovy
    def __init__(self, link='https://www.gutenberg.org/files/28054/28054-0.txt',
                 batch_size = 48,
                 batch_num = 3000,
                 sequence_length = 32):
        self.raw = request.urlopen(link).read().decode('utf8').lower()
        self.tokens = [c for c in self.raw]
        self.chars = sorted(set(self.tokens))
        self.inx2char = dict(enumerate(self.chars))
        self.char2inx = {ch: ii for ii, ch in self.inx2char.items()}
        self.encoded = [self.char2inx[inx] for inx in self.tokens]

        self.batch_size = batch_size
        self.batch_num = batch_num
        self.sequence_length = sequence_length
        # self.batch_num = len(self.encoded) // (self.batch_size * self.sequence_length)
        # self.encoded2arr = np.array(self.encoded[:self.batch_size * self.sequence_length * self.batch_num])
        # self.encoded2arr = self.encoded2arr.resize([self.batch_num * self.batch_size, self.sequence_length])

    def __len__(self):
        return self.batch_num

    def __getitem__(self, idx):
        batch_input = torch.LongTensor(self.batch_size, self.sequence_length)
        batch_output = torch.LongTensor(self.batch_size, self.sequence_length)

        for i in range(batch_input.shape[0]):
            start_idx = np.random.randint(0, len(self.encoded)-1-self.sequence_length)
            end_idx = start_idx + self.sequence_length
            batch_input[i] = torch.tensor(self.encoded[start_idx:end_idx])
            batch_output[i] = torch.tensor(self.encoded[start_idx+1:end_idx+1])
        return batch_input, batch_output
