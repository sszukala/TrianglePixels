# Cell 1: Install libraries
!pip install torchvision==0.15.2 tqdm==4.65.0 scipy==1.10.1 scikit-learn==1.3.0 psutil

# Cell 2: Import libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from scipy.ndimage import gaussian_filter, median_filter
import copy
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import torch.quantization
import time
import psutil
import random
from torch.utils.data import Dataset, DataLoader
import os  # Import os for memory usage calculations


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------
# --- Model Definitions ---
# ---------------------------------------------------

class TrianglePixelSplitter(nn.Module):
    def __init__(self):
        super(TrianglePixelSplitter, self).__init__()

    def forward(self, x):
        # Reshape the input tensor
        x = x.view(x.size(0), 1, x.shape[2], x.shape[3])

        # Create an empty tensor to store the split pixels
        split_pixels = torch.zeros(x.size(0), 2, x.shape[2], x.shape[3], device=x.device)

        # Split each square pixel into two triangles
        split_pixels[:, 0, ::2, ::2] = x[:, 0, ::2, ::2]
        split_pixels[:, 0, 1::2, 1::2] = x[:, 0, 1::2, 1::2]
        split_pixels[:, 1, ::2, 1::2] = x[:, 0, ::2, 1::2]
        split_pixels[:, 1, 1::2, ::2] = x[:, 0, 1::2, ::2]

        # Reshape back
        split_pixels = split_pixels.view(x.size(0), 2, x.shape[2], x.shape[3])

        return split_pixels

class Net(nn.Module):
    def __init__(self, input_channels=1):
        super(Net, self).__init__()
        self.triangle_splitter = TrianglePixelSplitter()
        self.conv1 = nn.Conv2d(input_channels * (2 if input_channels == 1 else 1), 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        fc1_input_size = 32 * 14 * 14 if input_channels == 1 else 32 * 16 * 16
        self.fc1 = nn.Linear(fc1_input_size, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(128, 10)  # Adjust output size if needed

    def forward(self, x):
        if x.shape[1] == 1:
            x = self.triangle_splitter(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class StudentNet(nn.Module):
    def __init__(self, input_channels=1):
        super(StudentNet, self).__init__()
        self.triangle_splitter = TrianglePixelSplitter()
        self.conv1 = nn.Conv2d(input_channels * (2 if input_channels == 1 else 1), 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        fc1_input_size = 16 * 14 * 14 if input_channels == 1 else 16 * 16 * 16
        self.fc1 = nn.Linear(fc1_input_size, 64)
        self.fc2 = nn.Linear(64, 10)  # Adjust output size if needed

    def forward(self, x):
        if x.shape[1] == 1:
            x = self.triangle_splitter(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# ---------------------------------------------------
# --- FashionMNIST Dataset Loading and Preprocessing ---
# ---------------------------------------------------

# Data transformations (adjust as needed)
transform = transforms.Compose([
    transforms.ToTensor(), # Convert PIL image to PyTorch tensor
    transforms.Normalize((0.5,), (0.5,)) # Normalize image data
])

# Create FashionMNIST dataset and dataloader
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                       download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=2)


# ---------------------------------------------------
# --- Model Initialization and Training ---
# ---------------------------------------------------

# Initialize teacher and student models
teacher_net = Net(input_channels=1).to(device)
student_net = StudentNet(input_channels=1).to(device)

# Loss function and optimizers
criterion = nn.CrossEntropyLoss()  # Using CrossEntropyLoss for FashionMNIST classification
optimizer_teacher = optim.Adam(teacher_net.parameters(), lr=0.001)
optimizer_student = optim.Adam(student_net.parameters(), lr=0.001)

# Training loop (teacher model)
num_epochs = 5  # Adjust as needed

for epoch in range(num_epochs):
    running_loss = 0.0
    with tqdm(trainloader, unit="batch", desc=f"Epoch {epoch + 1}/{num_epochs} (Teacher)") as tepoch:
        for i, data in enumerate(tepoch, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer_teacher.zero_grad()
            outputs = teacher_net(inputs)
            loss = criterion(outputs, labels)  
            loss.backward()
            optimizer_teacher.step()

            running_loss += loss.item()
            tepoch.set_postfix({"loss": f"{running_loss / (i + 1):.3f}"})
    print(f"Epoch {epoch + 1} (Teacher) - Loss: {running_loss / len(trainloader):.4f}")  # Print loss

# Training loop (student model)
for epoch in range(num_epochs):
    running_loss = 0.0
    with tqdm(trainloader, unit="batch", desc=f"Epoch {epoch + 1}/{num_epochs} (Student)") as tepoch:
        for i, data in enumerate(tepoch, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer_student.zero_grad()
            outputs = student_net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_student.step()

            running_loss += loss.item()
            tepoch.set_postfix({"loss": f"{running_loss / (i + 1):.3f}"})
    print(f"Epoch {epoch + 1} (Student) - Loss: {running_loss / len(trainloader):.4f}")  # Print loss

# ---------------------------------------------------
# --- Evaluation and Plotting ---
# ---------------------------------------------------

def measure_batch_time(model, dataloader, device):
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / (1024 ** 2)  # Memory in MB

    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            _ = model(inputs)
    end_time = time.time()
    total_time = end_time - start_time
    batches_per_second = len(dataloader) / total_time

    final_memory = process.memory_info().rss / (1024 ** 2)  # Memory in MB
    memory_used = final_memory - initial_memory
    print(f"Memory Used: {memory_used:.2f} MB")  # Print memory usage

    return batches_per_second

def measure_storage(model):
    total_size = 0
    for param in model.parameters():
        total_size += param.nelement() * param.element_size()
    return total_size

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate evaluation metrics
    accuracy = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)  # Added zero_division=0
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)  # Added zero_division=0
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)  # Added zero_division=0

    # Print metrics
    print(f'Accuracy: {accuracy:.2f}%')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

    return accuracy, precision, recall, f1

def plot_results(results, title, xlabel, ylabel):
    plt.figure(figsize=(8, 6))
    for model_name, data in results.items():
        plt.plot(data, label=model_name)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

# Measure batches/s for FashionMNIST data
batch_time_teacher = measure_batch_time(teacher_net, testloader, device)
batch_time_student = measure_batch_time(student_net, testloader, device)

# Measure memory for FashionMNIST data
square_pixel_model_size = measure_storage(teacher_net)
triangle_pixel_model_size = measure_storage(student_net)

# Evaluate on FashionMNIST data (using testloader)
_, _, _, f1_teacher = evaluate_model(teacher_net, testloader, device)
_, _, _, f1_student = evaluate_model(student_net, testloader, device)

# Plotting Batch/s Comparison
batch_times = {
    "Square Pixel Model": [batch_time_teacher],
    "Triangle Pixel Model": [batch_time_student]
}

memory_usage = {
    "Square Pixel Model": [square_pixel_model_size / (1024 * 1024)],
    "Triangle Pixel Model": [triangle_pixel_model_size / (1024 * 1024)]
}

f1_scores = {
    "Square Pixel Model": [f1_teacher],
    "Triangle Pixel Model": [f1_student]
}

plot_results(batch_times, "Batch/s Comparison (FashionMNIST)", "Model", "Batches/s")
plot_results(memory_usage, "Memory Usage Comparison (FashionMNIST)", "Model", "Memory (MB)")
plot_results(f1_scores, "F1 Score Comparison (FashionMNIST)", "Model", "F1 Score")

print("Training and evaluation completed!")
