import torch.nn as nn
import torch.nn.functional as F


class CNNSynchronizer(nn.Module):
    def __init__(self, input_length, num_classes):
        super(CNNSynchronizer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        
        # Max pooling layer
        self.pool = nn.MaxPool1d(2)

        # Dropout layer
        self.dropout = nn.Dropout(0.5)
        
        # Calculate the size after Conv1D layers and pooling
        conv1_out_size = input_length - (3 - 1)
        conv2_out_size = conv1_out_size - (3 - 1)
        conv3_out_size = (conv2_out_size - (3 - 1)) // 2
        conv4_out_size = (conv3_out_size - (3 - 1)) // 2
        
        self.fc1 = nn.Linear(conv4_out_size * 128, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

