"""===========================================================================================
Required Libraries
==========================================================================================="""
import os
import shutil

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from dataset import ShapeDataset, classes, image_size
from tqdm import tqdm



"""===========================================================================================
Settings
==========================================================================================="""
batch_size = 16

dir_proj = os.getcwd()
dir_dataset = os.path.join(dir_proj, "013_classification_transfer_learning", "dataset")
path_model = os.path.join(dir_proj, "013_classification_transfer_learning",
                          "models", "resnet18_transfer_shapes.pt")
dir_output = os.path.join(dir_proj, "013_classification_transfer_learning", "output")


"""===========================================================================================
Pretrained Model
==========================================================================================="""
def make_model():
    model = resnet18(weights=None)
    model.fc = nn.Linear(in_features=model.fc.in_features,
                         out_features=len(classes))
    return model


"""===========================================================================================
Helper Functions
==========================================================================================="""
def draw_prediction(path, pred_name):
    img = cv2.imread(filename=path)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9
    thickness = 2
    (_, text_h), baseline = cv2.getTextSize(text=pred_name,
                                            fontFace=font,
                                            fontScale=scale,
                                            thickness=thickness)
    y = min(image_size - baseline - 6, text_h + 10)

    cv2.putText(img=img, text=pred_name, org=(8, y),
                fontFace=font, fontScale=scale,
                color=(0, 0, 255), thickness=thickness,
                lineType=cv2.LINE_AA)
    return img


"""===========================================================================================
Load Model and Predict
==========================================================================================="""
device = "cuda" if torch.cuda.is_available() else "cpu"

test_dataset = ShapeDataset(dir_split=os.path.join(dir_dataset, "test"),
                            return_path=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

model = make_model().to(device=device)
model.load_state_dict(state_dict=torch.load(f=path_model,
                                            map_location=device))
model.eval()

if os.path.exists(dir_output):
    shutil.rmtree(dir_output)
os.makedirs(dir_output, exist_ok=True)

y_true = []
y_pred = []

for images, labels, paths in tqdm(iterable=test_loader, desc="Predicting"):
    images = images.to(device=device)

    with torch.no_grad():
        outputs = model(images)
        pred_idxs = torch.argmax(input=outputs, dim=1).cpu().numpy()

    for label, pred_idx, path in zip(labels.numpy(), pred_idxs, paths):
        pred_name = classes[int(pred_idx)]
        y_true.append(int(label))
        y_pred.append(int(pred_idx))

        result = draw_prediction(path=path, pred_name=pred_name)
        cv2.imwrite(filename=os.path.join(dir_output, os.path.basename(path)),
                    img=result)


"""===========================================================================================
Print Metrics
==========================================================================================="""
confusion = np.zeros(shape=(len(classes), len(classes)), dtype=np.int32)

for true_idx, pred_idx in zip(y_true, y_pred):
    confusion[true_idx, pred_idx] += 1

accuracy = np.trace(confusion) / np.sum(confusion)
precision_sum = 0
recall_sum = 0
f1_sum = 0

for idx in range(len(classes)):
    tp = confusion[idx, idx]
    fp = np.sum(confusion[:, idx]) - tp
    fn = np.sum(confusion[idx, :]) - tp

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) \
        if precision + recall > 0 else 0

    precision_sum += precision
    recall_sum += recall
    f1_sum += f1

print("\nClassification metrics")
print("Accuracy:", round(accuracy, 3))
print("Precision:", round(precision_sum / len(classes), 3))
print("Recall:", round(recall_sum / len(classes), 3))
print("F1-score:", round(f1_sum / len(classes), 3))
print("Saved prediction images to:", dir_output)
