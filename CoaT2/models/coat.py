import torch.nn as nn
from .cait import CAIT
from .cfan import CFAN

class CoaT(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CoaT, self).__init__()
        self.cait = CAIT(in_channels, num_classes)
        self.cfan = CFAN(num_classes)

    def forward(self, x):
        x = self.cait(x)
        x = self.cfan(x)
        return x

class CoaTObjectDetector(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CoaTObjectDetector, self).__init__()
        self.coat = CoaT(in_channels, num_classes)
        self.bbox_head = nn.Conv2d(num_classes, 4, kernel_size=1)
        self.class_head = nn.Conv2d(num_classes, 1, kernel_size=1)

    def forward(self, x):
        features = self.coat(x)
        bbox_preds = self.bbox_head(features)
        class_preds = self.class_head(features)
        return bbox_preds, class_preds
