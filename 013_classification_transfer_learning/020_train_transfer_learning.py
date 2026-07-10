"""===========================================================================================
Required Libraries
==========================================================================================="""
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

from dataset import ShapeDataset, classes
from tqdm import tqdm


"""===========================================================================================
Settings
==========================================================================================="""
batch_size = 16
num_epochs = 10

dir_proj = os.getcwd()
dir_dataset = os.path.join(dir_proj, "013_classification_transfer_learning", "dataset")
dir_model = os.path.join(dir_proj, "013_classification_transfer_learning", "models")
path_model = os.path.join(dir_model, "resnet18_transfer_shapes.pt")

torch.manual_seed(seed=13)


"""===========================================================================================
Pretrained Model
==========================================================================================="""
def make_model():
    # ResNet18 is already trained on ImageNet. We replace only the last classifier layer.
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    print(model)
    model.fc = nn.Linear(in_features=model.fc.in_features,
                         out_features=len(classes))
    return model


def freeze_backbone(model):
    # Transfer learning: keep the pretrained CNN feature extractor fixed.
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True


"""===========================================================================================
Train and Evaluate
==========================================================================================="""
def get_accuracy(loader, desc):
    model.eval()
    correct = 0
    total = 0

    for images, labels in tqdm(iterable=loader, desc=desc):
        images = images.to(device=device)
        labels = labels.to(device=device)

        with torch.no_grad():
            outputs = model(images)

        pred = torch.argmax(input=outputs, dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(dim=0)

    return correct / total


def train_epoch(loader, desc, optimizer):
    model.train()
    correct = 0
    total = 0

    for images, labels in tqdm(iterable=loader, desc=desc):
        images = images.to(device=device)
        labels = labels.to(device=device)

        outputs = model(images)
        loss = loss_fn(input=outputs, target=labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = torch.argmax(input=outputs, dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(dim=0)

    return correct / total


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

train_dataset = ShapeDataset(dir_split=os.path.join(dir_dataset, "train"))
valid_dataset = ShapeDataset(dir_split=os.path.join(dir_dataset, "valid"))
test_dataset = ShapeDataset(dir_split=os.path.join(dir_dataset, "test"))

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                          shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

model = make_model().to(device=device)
loss_fn = nn.CrossEntropyLoss()

os.makedirs(dir_model, exist_ok=True)
best_valid_acc = -1

before_acc = get_accuracy(loader=test_loader, desc="Before transfer learning")
print("Before transfer learning test accuracy:", round(before_acc, 4))

print("\nTransfer learning: train only the new classifier layer")
freeze_backbone(model)
optimizer = torch.optim.Adam(params=model.fc.parameters(), lr=0.001)

for epoch in range(1, num_epochs + 1):
    train_acc = train_epoch(loader=train_loader,
                            desc=f"Transfer {epoch}/{num_epochs}",
                            optimizer=optimizer)
    valid_acc = get_accuracy(loader=valid_loader,
                             desc=f"Valid {epoch}/{num_epochs}")
    print(f"Transfer {epoch}/{num_epochs} | "
          f"train acc {train_acc:.3f} | valid acc {valid_acc:.3f}")

    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        torch.save(obj=model.state_dict(), f=path_model)

model.load_state_dict(state_dict=torch.load(f=path_model,
                                            map_location=device))
after_acc = get_accuracy(loader=test_loader, desc="After transfer learning")

print("\nBefore transfer learning accuracy:", round(before_acc, 4))
print("After transfer learning accuracy:", round(after_acc, 4))
print("Saved model to:", path_model)
