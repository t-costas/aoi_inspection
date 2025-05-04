import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision.models import ResNet18_Weights
from torchvision import datasets
from torch.utils.data.sampler import WeightedRandomSampler
import datetime
# === Optional fix for OpenMP error on Windows ===
#os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
print("Starting...")

# === Configuration ===
data_dir = "dataset"
batch_size = 8
num_epochs = 10
learning_rate = 6e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Image Preprocessing ===
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),       # Randomly flip images horizontally
    transforms.RandomVerticalFlip(),         # Randomly flip images vertically
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# === Dataset Loading ===
# For training set, apply both oversampling and augmentation
train_dataset = torchvision.datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)

# Calculate class counts for oversampling
class_counts = [0, 0]
for _, label in train_dataset:
    class_counts[label] += 1

# Class weights for oversampling
total_samples = sum(class_counts)
weights = [total_samples / class_counts[label] for label in range(len(class_counts))]

# Assign weights to each sample
sample_weights = [weights[label] for _, label in train_dataset]

# Create sampler for oversampling
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

# DataLoader with sampler and augmentation
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

# Validation set
val_dataset = torchvision.datasets.ImageFolder(os.path.join(data_dir, "val"), transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# === Model Setup ===
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)  # Binary classification
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# === Training Loop ===
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

# Track misclassified images
misclassified_images = []
misclassified_labels = []
misclassified_preds = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train, total_train = 0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct_train += (preds == labels).sum().item()
        total_train += labels.size(0)

    avg_train_loss = running_loss / len(train_loader)
    train_acc = correct_train / total_train
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_acc)

    # === Validation ===
    model.eval()
    val_loss = 0.0
    correct_val, total_val = 0, 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)

            # Track misclassified boards
            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    misclassified_images.append(inputs[i].cpu())  # Move to CPU for easier handling
                    misclassified_labels.append(labels[i].cpu().numpy())
                    misclassified_preds.append(preds[i].cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_acc = correct_val / total_val
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}] | "
          f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2%} | "
          f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2%}")

# === Save Model with Current Date ===
current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_save_path = f"model_{current_date}.pth"

# Save the model state
torch.save(model.state_dict(), model_save_path)
print(f"Model saved at {model_save_path}")

# === Plot Loss Curves ===
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss", marker='o')
plt.plot(val_losses, label="Val Loss", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot Accuracy Curves ===
plt.figure(figsize=(8, 5))
plt.plot(train_accuracies, label="Train Accuracy", marker='o')
plt.plot(val_accuracies, label="Val Accuracy", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Confusion Matrix ===
all_preds, all_labels = [], []
model.eval()
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_dataset.classes)

plt.figure(figsize=(6, 5))
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix on Validation Set")
plt.tight_layout()
plt.show()

# === Visualizing Misclassified Boards ===
for i in range(min(5, len(misclassified_images))):  # Show first 5 misclassified images
    img = transforms.ToPILImage()(misclassified_images[i])  # Convert tensor to image
    plt.imshow(img)
    plt.title(f"True: {misclassified_labels[i]}, Pred: {misclassified_preds[i]}")
    plt.show()