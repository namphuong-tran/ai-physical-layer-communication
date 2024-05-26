import torch
import torch.nn as nn
import torch.nn.functional as F

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
