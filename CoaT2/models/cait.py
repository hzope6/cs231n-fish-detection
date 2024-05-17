import torch
import torch.nn as nn

class CAIT(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CAIT, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.transformer = nn.Transformer(d_model=64, nhead=8, num_encoder_layers=6)
        self.conv2 = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = x.flatten(2).permute(2, 0, 1)  # Prepare for transformer
        x = self.transformer(x)
        x = x.permute(1, 2, 0).view(-1, 64, x.size(0)//64, x.size(0)//64)
        x = self.conv2(x)
        return x
