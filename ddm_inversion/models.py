from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
from torch.amp.autocast_mode import autocast
import torch
from torch import nn


def decode_latents(
    model: StableDiffusionPipeline, w0, mixed_precision=True
) -> torch.Tensor:
    # if mixed_precision:
    #     with autocast("cuda"):
    #         x0_decoded = model.vae.decode(1 / 0.18215 * w0).sample
    # else:
    x0_decoded = model.vae.decode(1 / 0.18215 * w0).sample
    return x0_decoded


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
