import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class CNNPacketDetection(nn.Module):
    def __init__(self, input_length, num_classes):
        super(CNNPacketDetection, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=9, kernel_size=3, stride=1)
        self.conv2 = nn.Conv1d(in_channels=9, out_channels=5, kernel_size=3, stride=1)
        self.flattened_size = 5 * (input_length - 4)  
        self.fc1 = nn.Linear(self.flattened_size, 3)
        self.fc2 = nn.Linear(3, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the datasets from files
dataset_y = np.load('dataset_y.npy')
dataset_t = np.load('dataset_t.npy')

# Create a custom PyTorch Dataset class
class TimingDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

# Create the dataset and DataLoader
timing_dataset = TimingDataset(dataset_y, dataset_t)
dataloader = DataLoader(timing_dataset, batch_size=32, shuffle=True)

M = dataset_y.shape[1]
N_u = timing_dataset.shape[1]
# Define the input length and number of classes based on your data
input_length = 2 * M  # Example input length, adjust based on your actual data
num_classes = N_u  
# Initialize the model, define the loss function and the optimizer
model = CNNPacketDetection(input_length, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        # Ensure the inputs are of shape (batch_size, 1, input_length)
        inputs = inputs.unsqueeze(1)

        # Convert labels to appropriate format for CrossEntropyLoss
        labels = torch.argmax(labels, dim=1)
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
