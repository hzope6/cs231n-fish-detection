import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class YoloDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_list = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        label_path = os.path.join(self.label_dir, self.img_list[idx].replace('.jpg', '.txt'))
        
        img = Image.open(img_path).convert('RGB')
        boxes = self.read_yolo_label(label_path)
        
        if self.transform:
            img = self.transform(img)
        
        return img, boxes

    def read_yolo_label(self, label_path):
        boxes = []
        with open(label_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                label = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
                boxes.append([label, x_center, y_center, width, height])
        return torch.tensor(boxes)

def get_dataloader(img_dir, label_dir, batch_size=4, transform=None):
    dataset = YoloDataset(img_dir, label_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
