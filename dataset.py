import torch
import random

class StoryDataset:
    def __init__(self, token_sequences, block_size=32):
        self.sequences = token_sequences
        self.block_size = block_size

    def get_batch(self, batch_size=32):
        X = []
        Y = []

        for _ in range(batch_size):
            seq = random.choice(self.sequences)
            start = random.randint(0, len(seq) - self.block_size - 1)

            x = seq[start : start + self.block_size]
            y = seq[start + 1 : start + self.block_size + 1]

            X.append(x)
            Y.append(y)

        return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)
