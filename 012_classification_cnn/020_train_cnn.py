"""===========================================================================================
Required Libraries
==========================================================================================="""
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import ShapeDataset, classes
from tqdm import tqdm


"""===========================================================================================
Settings
==========================================================================================="""
batch_size = 32
num_epochs = 10

dir_proj = os.getcwd()
dir_dataset = os.path.join(dir_proj, "012_classification_cnn", "dataset")
dir_model = os.path.join(dir_proj, "012_classification_cnn", "models")
path_model = os.path.join(dir_model, "cnn_shapes.pt")

torch.manual_seed(seed=10)


"""===========================================================================================
CNN Model
==========================================================================================="""
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=64 * 12 * 12, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=len(classes)),
        )

    def forward(self, x):
        return self.net(x)


"""===========================================================================================
Train and Evaluate
==========================================================================================="""
def run_epoch(loader, desc, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    correct = 0
    total = 0

    for images, labels in tqdm(iterable=loader, desc=desc):
        images = images.to(device=device)
        labels = labels.to(device=device)

        with torch.set_grad_enabled(mode=is_train):
            outputs = model(images)
            loss = loss_fn(input=outputs, target=labels)

            if is_train:
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

model = SmallCNN().to(device=device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

os.makedirs(dir_model, exist_ok=True)
best_valid_acc = -1

for epoch in range(1, num_epochs + 1):
    train_acc = run_epoch(loader=train_loader,
                          desc=f"Train {epoch}/{num_epochs}",
                          optimizer=optimizer)
    valid_acc = run_epoch(loader=valid_loader,
                          desc=f"Valid {epoch}/{num_epochs}")
    print(f"Epoch {epoch}/{num_epochs} | "
          f"train acc {train_acc:.3f} | valid acc {valid_acc:.3f}")

    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        torch.save(obj=model.state_dict(), f=path_model)

model.load_state_dict(state_dict=torch.load(f=path_model,
                                            map_location=device))
test_acc = run_epoch(loader=test_loader, desc="Test")

print("Test accuracy:", round(test_acc, 4))
print("Saved model to:", path_model)
