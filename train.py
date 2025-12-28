import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from tqdm import tqdm
from medical_resnet import MedicalCNN
from utils import get_class_weights, evaluate_model, save_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_path = r"D:\chest_xray\chest_xray\train"
test_path = r"D:\chest_xray\chest_xray\test"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(train_path, transform=transform)
train_len = int(0.8 * len(dataset))
val_len = len(dataset) - train_len
train_ds, val_ds = random_split(dataset, [train_len, val_len])

class_weights = get_class_weights(dataset, device)
targets = [train_ds[i][1] for i in range(len(train_ds))]
weights = [class_weights[t].item() for t in targets]

sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler)
val_loader = DataLoader(val_ds, batch_size=32)

model = MedicalCNN().to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

best_loss = float("inf")
patience, counter = 7, 0

for epoch in range(10):
    model.train()
    for x, y in tqdm(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    val_loss, val_acc, _, _ = evaluate_model(
        model, val_loader, criterion, device
    )

    print(f"Epoch {epoch+1} | Val Acc: {val_acc*100:.2f}%")

    if val_loss < best_loss:
        best_loss = val_loss
        counter = 0
        save_model(model, "best_model.pth")
    else:
        counter += 1
        if counter >= patience:
            break
