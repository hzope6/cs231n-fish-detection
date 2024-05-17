import torch
from data.prepare_data import get_dataloader, transform
from models.coat import CoaTObjectDetector

def evaluate():
    img_dir = 'data/images'
    label_dir = 'data/labels'
    dataloader = get_dataloader(img_dir, label_dir, transform=transform)

    model = CoaTObjectDetector(in_channels=3, num_classes=512).cuda()
    model.load
