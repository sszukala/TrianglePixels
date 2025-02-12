# Cell 1: Install libraries
!pip install torchvision==0.15.2
!pip install tqdm==4.65.0
!pip install scipy==1.10.1
!pip install scikit-learn==1.3.0
!pip install psutil

# Cell 2: Import libraries
from IPython import get_ipython
from IPython.display import display
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import gaussian_filter, median_filter
import copy
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import torch.quantization
import time
import psutil

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define a custom layer to split square pixels into triangular pixels
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

# Define the teacher model (original Net)
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
        self.fc2 = nn.Linear(128, 10)

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

# Define the student model (smaller version of Net)
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
        self.fc2 = nn.Linear(64, 10)

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

# Define functions for image transformations
def gaussian_blur(image, sigma=1.0):
    blurred_image = gaussian_filter(image.numpy(), sigma=sigma)
    return torch.from_numpy(blurred_image)

def median_blur(image, size=5):
    blurred_image = median_filter(image.numpy(), size=size)
    return torch.from_numpy(blurred_image)

def equalize_hist(image):
    import torchvision.transforms.functional as TF
    image_pil = TF.to_pil_image(image)
    equalized_image_pil = TF.equalize(image_pil)
    equalized_image = TF.to_tensor(equalized_image_pil)
    return equalized_image


# ----- Memory Efficiency Experiment (FashionMNIST) -----
# Data loading and preprocessing
transform_fashion = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: gaussian_blur(x, sigma=1.0)),
    transforms.Lambda(lambda x: median_blur(x, size=5)),
    transforms.Lambda(lambda x: equalize_hist(x)),
])

trainset_fashion = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                        download=True, transform=transform_fashion)
testset_fashion = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                       download=True, transform=transform_fashion)

# Create smaller datasets
subset_size_fashion = 700
train_subset_indices_fashion = torch.randperm(len(trainset_fashion))[:subset_size_fashion]
test_subset_indices_fashion = torch.randperm(len(testset_fashion))[:subset_size_fashion]

train_subset_fashion = torch.utils.data.Subset(trainset_fashion, train_subset_indices_fashion)
test_subset_fashion = torch.utils.data.Subset(testset_fashion, test_subset_indices_fashion)

# Create DataLoaders with batch size 200
trainloader_fashion = torch.utils.data.DataLoader(train_subset_fashion, batch_size=200,
                                          shuffle=True, num_workers=2)
testloader_fashion = torch.utils.data.DataLoader(test_subset_fashion, batch_size=200,
                                         shuffle=False, num_workers=2)

# Initialize and train models for memory efficiency experiment
teacher_net_fashion = Net().to(device)
student_net_fashion = StudentNet().to(device)

# Training loop (FashionMNIST - Memory Efficiency)
def train_model(model, trainloader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        running_loss = 0.0
        with tqdm(trainloader, unit="batch", desc=f"Epoch {epoch + 1}/{num_epochs}") as tepoch:
            for i, data in enumerate(tepoch, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                tepoch.set_postfix({"loss": f"{running_loss / (i + 1):.3f}"})
                if i % 200 == 199:
                    running_loss = 0.0

criterion = nn.CrossEntropyLoss()
optimizer_teacher_fashion = optim.Adam(teacher_net_fashion.parameters(), lr=0.001, weight_decay=0.001)
optimizer_student_fashion = optim.Adam(student_net_fashion.parameters(), lr=0.001, weight_decay=0.001)

num_epochs_fashion = 5

train_model(teacher_net_fashion, trainloader_fashion, criterion, optimizer_teacher_fashion, num_epochs_fashion)
train_model(student_net_fashion, trainloader_fashion, criterion, optimizer_student_fashion, num_epochs_fashion)


# ----- Visual Perception Experiment (CIFAR-10) -----
# Data loading and preprocessing
transform_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset_cifar = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_cifar)
testset_cifar = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_cifar)

# Create smaller datasets
subset_size_cifar = 700
train_subset_indices_cifar = torch.randperm(len(trainset_cifar))[:subset_size_cifar]
test_subset_indices_cifar = torch.randperm(len(testset_cifar))[:subset_size_cifar]

train_subset_cifar = torch.utils.data.Subset(trainset_cifar, train_subset_indices_cifar)
test_subset_cifar = torch.utils.data.Subset(testset_cifar, test_subset_indices_cifar)

# Create DataLoaders with batch size 200
trainloader_cifar = torch.utils.data.DataLoader(train_subset_cifar, batch_size=200,
                                          shuffle=True, num_workers=2)
testloader_cifar = torch.utils.data.DataLoader(test_subset_cifar, batch_size=200,
                                         shuffle=False, num_workers=2)

# Create validation set for CIFAR-10
train_size_cifar = int(0.8 * len(train_subset_cifar))
val_size_cifar = len(train_subset_cifar) - train_size_cifar
train_dataset_cifar, val_dataset_cifar = torch.utils.data.random_split(
    train_subset_cifar, [train_size_cifar, val_size_cifar]
)

valloader_cifar = torch.utils.data.DataLoader(val_dataset_cifar, batch_size=200,
                                         shuffle=False, num_workers=2)

# Initialize and train models for visual perception experiment
teacher_net_cifar = Net(input_channels=3).to(device)
student_net_cifar = StudentNet(input_channels=3).to(device)

# Training loop (CIFAR-10 - Visual Perception)
criterion = nn.CrossEntropyLoss()
optimizer_teacher_cifar = optim.Adam(teacher_net_cifar.parameters(), lr=0.001, weight_decay=0.001)
optimizer_student_cifar = optim.Adam(student_net_cifar.parameters(), lr=0.001, weight_decay=0.001)

num_epochs_cifar = 5

train_model(teacher_net_cifar, trainloader_cifar, criterion, optimizer_teacher_cifar, num_epochs_cifar)
train_model(student_net_cifar, trainloader_cifar, criterion, optimizer_student_cifar, num_epochs_cifar)


# --- Plotting Results ---
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

# --- Batch/s Comparison ---
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

# Measure batches/s for FashionMNIST
batch_time_teacher_fashion = measure_batch_time(teacher_net_fashion, testloader_fashion, device)
batch_time_student_fashion = measure_batch_time(student_net_fashion, testloader_fashion, device)

# Measure batches/s for CIFAR-10
batch_time_teacher_cifar = measure_batch_time(teacher_net_cifar, testloader_cifar, device)
batch_time_student_cifar = measure_batch_time(student_net_cifar, testloader_cifar, device)

# Plotting Batch/s Comparison
batch_times_fashion = {
    "Square Pixel Model": [batch_time_teacher_fashion],
    "Triangle Pixel Model": [batch_time_student_fashion]
}
batch_times_cifar = {
    "Square Pixel Model": [batch_time_teacher_cifar],
    "Triangle Pixel Model": [batch_time_student_cifar]
}

plot_results(batch_times_fashion, "Batch/s Comparison (FashionMNIST)", "Model", "Batches/s")
plot_results(batch_times_cifar, "Batch/s Comparison (CIFAR-10)", "Model", "Batches/s")

# --- Memory Usage ---
def measure_storage(model):
    total_size = 0
    for param in model.parameters():
        total_size += param.nelement() * param.element_size()
    return total_size

# Measure memory for FashionMNIST
square_pixel_model_size_fashion = measure_storage(teacher_net_fashion)
triangle_pixel_model_size_fashion = measure_storage(student_net_fashion)

# Measure memory for CIFAR-10
square_pixel_model_size_cifar = measure_storage(teacher_net_cifar)
triangle_pixel_model_size_cifar = measure_storage(student_net_cifar)

# Plotting Memory Usage
memory_usage_fashion = {
    "Square Pixel Model": [square_pixel_model_size_fashion / (1024 * 1024)],
    "Triangle Pixel Model": [triangle_pixel_model_size_fashion / (1024 * 1024)]
}
memory_usage_cifar = {
    "Square Pixel Model": [square_pixel_model_size_cifar / (1024 * 1024)],
    "Triangle Pixel Model": [triangle_pixel_model_size_cifar / (1024 * 1024)]
}

plot_results(memory_usage_fashion, "Memory Usage Comparison (FashionMNIST)", "Model", "Memory (MB)")
plot_results(memory_usage_cifar, "Memory Usage Comparison (CIFAR-10)", "Model", "Memory (MB)")

# Evaluate on validation set
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

print("Evaluating Square Pixel Model (Net) on CIFAR-10 validation set...")
evaluate_model(teacher_net_cifar, valloader_cifar, device)

print("Evaluating Triangle Pixel Model (StudentNet) on CIFAR-10 validation set...")
evaluate_model(student_net_cifar, valloader_cifar, device)

# Knowledge Distillation
temperature = 2.0
alpha = 0.5

def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    kd_loss = nn.KLDivLoss()(F.log_softmax(student_logits/T, dim=1),
                             F.softmax(teacher_logits/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(student_logits, labels) * (1. - alpha)
    return kd_loss

# Early Stopping parameters
patience = 5
best_val_loss = float('inf')
epochs_without_improvement = 0
best_model_wts = copy.deepcopy(student_net_cifar.state_dict())

# Training loop with Knowledge Distillation and Early Stopping for CIFAR-10
for epoch in range(num_epochs_cifar):
    running_loss = 0.0
    with tqdm(trainloader_cifar, unit="batch", desc=f"Epoch {epoch + 1}/{num_epochs_cifar}") as tepoch:
        for i, data in enumerate(tepoch, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer_student_cifar.zero_grad()

            teacher_outputs = teacher_net_cifar(inputs)
            student_outputs = student_net_cifar(inputs)

            loss = distillation_loss(student_outputs, teacher_outputs, labels, temperature, alpha)

            loss.backward()
            optimizer_student_cifar.step()
            running_loss += loss.item()
            tepoch.set_postfix({"loss": f"{running_loss / (i + 1):.3f}"})
            if i % 200 == 199:
                running_loss = 0.0

    # Validation at the end of each epoch
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in valloader_cifar:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = student_net_cifar(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(valloader_cifar)
    val_accuracy = 100 * correct / total
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        best_model_wts = copy.deepcopy(student_net_cifar.state_dict())
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            student_net_cifar.load_state_dict(best_model_wts)
            break

# Visualization code 
def visualize_pixels(image, pixel_type):
    """Visualizes an image with highlighted pixel structure.

    Args:
        image (torch.Tensor): The input image.
        pixel_type (str): 'triangle' or 'square'.

    Returns:
        torch.Tensor: The visualized image.
    """

    image = image.cpu().numpy()  # Convert to NumPy array
    image = np.squeeze(image)  # Remove channel dimension if present

    if pixel_type == 'triangle':
        # Highlight triangle pixel boundaries
        image[::2, 1::2] = 0  # Darken alternate rows for triangle effect
        image[1::2, ::2] = 0  # Darken alternate columns for triangle effect
    elif pixel_type == 'square':
        # Highlight square pixel boundaries
        # (No specific modification needed, as square pixels are the default)
        pass
    else:
        raise ValueError("Invalid pixel_type. Choose 'triangle' or 'square'.")

