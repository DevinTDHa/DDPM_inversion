import torch
from torch import nn


class BaselineColorPredictor(nn.Module):
    CHANNEL_RED = 0
    CHANNEL_GREEN = 1
    CHANNEL_BLUE = 2

    def __init__(self, channel=CHANNEL_RED):
        super(BaselineColorPredictor, self).__init__()
        self.channel = channel

    def forward(self, x):
        assert x.dim() == 4, "Input tensor must be 4D (batch, channel, height, width)"
        return x[:, self.channel, ...].mean(dim=(1, 2))
