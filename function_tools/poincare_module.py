import torch
from torch import nn
from torch.autograd import Function

class PoincareEmbedding(nn.Module):
    def __init__(self, N, M=3):
        super(PoincareEmbedding, self).__init__()
        # because precision can cause go out of the circle bound
        self.l_embed = nn.Embedding(N, M, max_norm=1-1e-4)
        self.l_embed.weight.data[:,:] = self.l_embed.weight.data[:,:] * 1e-2
    def forward(self, x):
        return self.l_embed(x)
