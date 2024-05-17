import torch
import torch.nn as nn

class SerialBlock(nn.Module):
    def __init__(self, in_channels):
        super(SerialBlock, self).__init__()
        self.downsample = nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=2, padding=1)
        self.conv_attention = nn.Transformer(d_model=in_channels * 2, nhead=8, num_encoder_layers=6)
    
    def forward(self, x):
        x = self.downsample(x)
        x = x.flatten(2).permute(2, 0, 1)  # Prepare for transformer
        x = self.conv_attention(x)
        x = x.permute(1, 2, 0).view(-1, x.size(1), x.size(0)//x.size(1), x.size(0)//x.size(1))
        return x

class ParallelBlock(nn.Module):
    def __init__(self, in_channels):
        super(ParallelBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        return x1 + x2

class CFAN(nn.Module):
    def __init__(self, in_channels):
        super(CFAN, self).__init__()
        self.serial_block = SerialBlock(in_channels)
        self.parallel_block = ParallelBlock(in_channels)

    def forward(self, x):
        x = self.serial_block(x)
        x = self.parallel_block(x)
        return x
