import torch
import torch.optim as optim
import torch.nn.functional as F
from data.prepare_data import get_dataloader, transform
from models.coat import CoaTObjectDetector

def compute_loss(pred_boxes, pred_classes, true_boxes):
    bbox_loss = F.mse_loss(pred_boxes, true_boxes[:, :, :4])
    class_loss = F.binary_cross_entropy_with_logits(pred_classes, true_boxes[:, :, 4:])
    return bbox_loss + class_loss

def train():
    img_dir = './data/Ozfish-dataset/train/images'
    label_dir = './data/Ozfish-dataset/train/labels'
    dataloader = get_dataloader(img_dir, label_dir, transform=transform)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CoaTObjectDetector(in_channels=3, num_classes=512).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for imgs, true_boxes in dataloader:
            imgs = imgs.cuda()
            true_boxes = true_boxes.cuda()

            optimizer.zero_grad()
            pred_boxes, pred_classes = model(imgs)
            
            loss = compute_loss(pred_boxes, pred_classes, true_boxes)
            print(loss)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader)}")

    torch.save(model.state_dict(), "coat_object_detector.pth")

if __name__ == "__main__":
    train()
