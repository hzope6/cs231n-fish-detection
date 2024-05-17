import torch.nn as nn

class MLP(nn.Sequential):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.add_module("conv1", nn.Conv2d(n_in, n_in, 1, 1))
        self.add_module("ln1", nn.LayerNorm([n_in, 1, 1]))
        self.add_module("relu", nn.ReLU(True))
        self.add_module("conv2", nn.Conv2d(n_in, n_out, 1, 1))
