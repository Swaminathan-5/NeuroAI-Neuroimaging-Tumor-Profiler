import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from PIL import Image

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
DATASET_DIR = 'brain_tumor_dataset'
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20
VAL_SPLIT = 0.2

# Stronger data augmentation for training
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset with new transforms
full_dataset = datasets.ImageFolder(DATASET_DIR, transform=train_transform)
val_dataset_for_split = datasets.ImageFolder(DATASET_DIR, transform=val_transform)
class_names = full_dataset.classes

val_size = int(len(full_dataset) * VAL_SPLIT)
train_size = len(full_dataset) - val_size
train_dataset, _ = torch.utils.data.random_split(full_dataset, [train_size, val_size])
_, val_dataset = torch.utils.data.random_split(val_dataset_for_split, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Transfer learning with ResNet18
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for param in resnet.parameters():
    param.requires_grad = False  # Freeze all layers

# Enable fine-tuning of the last ResNet block
for param in resnet.layer4.parameters():
    param.requires_grad = True

# Replace the final layer for binary classification
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Sequential(
    nn.Linear(num_ftrs, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 1)
)
resnet = resnet.to(DEVICE)

# Only train the final layer and last block
optimizer = optim.Adam(filter(lambda p: p.requires_grad, resnet.parameters()), lr=0.0005)
criterion = nn.BCEWithLogitsLoss()

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# Early stopping parameters
early_stopping_patience = 5
best_val_loss = float('inf')
epochs_no_improve = 0
best_model_state = None

# Training and evaluation functions (same as before, but use 'resnet' instead of 'model')
def train(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.float().to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.squeeze(1), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        preds = torch.sigmoid(outputs.squeeze(1)) > 0.5
        correct += (preds == labels.bool()).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total

def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.float().to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs.squeeze(1), labels)
            running_loss += loss.item() * images.size(0)
            preds = torch.sigmoid(outputs.squeeze(1)) > 0.5
            correct += (preds == labels.bool()).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total

train_losses, val_losses, train_accs, val_accs = [], [], [], []
for epoch in range(EPOCHS):
    train_loss, train_acc = train(resnet, train_loader, criterion, optimizer)
    val_loss, val_acc = evaluate(resnet, val_loader, criterion)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    scheduler.step(val_loss)
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = resnet.state_dict()
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Restore best model
if best_model_state is not None:
    resnet.load_state_dict(best_model_state)

# Plot training history
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.title('Accuracy')
plt.legend()
plt.subplot(1,2,2)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Loss')
plt.legend()
plt.show()

# Final evaluation
val_loss, val_acc = evaluate(resnet, val_loader, criterion)
print(f'Final Validation Accuracy: {val_acc:.4f}')

# Save the trained model
MODEL_PATH = 'brain_tumor_resnet18.pth'
torch.save(resnet.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# Confusion matrix
def get_all_preds(model, loader):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = (torch.sigmoid(outputs.squeeze(1)) > 0.5).cpu().numpy().astype(int)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy().astype(int))
    return np.array(all_labels), np.array(all_preds)

val_labels, val_preds = get_all_preds(resnet, val_loader)
cm = confusion_matrix(val_labels, val_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix (Validation)')
plt.show()

# Prediction function for new images (use resnet)
def predict_image(image_path, model, device=DEVICE):
    model.eval()
    img = Image.open(image_path).convert('RGB')
    img = val_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
        prob = torch.sigmoid(output).item()
        pred = int(prob > 0.5)
    print(f"Prediction: {class_names[pred]} (probability: {prob:.4f})")
    return class_names[pred], prob

# ---
# Overfitting/Underfitting Guidance:
# - If training accuracy is much higher than validation accuracy, try:
#   * More data augmentation
#   * Increasing dropout
#   * Enabling fine-tuning of the last ResNet block (see above)
#   * Early stopping or reducing learning rate
# - If both accuracies are low, try:
#   * Training for more epochs
#   * Unfreezing more layers for fine-tuning
#   * Checking data quality and class balance
# --- 