from dataset import ChunkedTimingDataset
import glob
from torch.utils.data import DataLoader
from synchronizer import CNNSynchronizer
import torch

# Function to load files
def load_files(prefix):
    y_files = sorted(glob.glob(f'{prefix}/{prefix}_y_*.npy'))
    t_files = sorted(glob.glob(f'{prefix}/{prefix}_t_*.npy'))
    return y_files, t_files

# Load the training and evaluation datasets from files
eval_y_files, eval_t_files = load_files('eval')

eval_dataset = ChunkedTimingDataset(eval_y_files, eval_t_files)

eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

# Load the model
input_length = eval_dataset.y_data.shape[1]  
num_classes = eval_dataset.t_data.shape[1]   
model = CNNSynchronizer(input_length, num_classes)
model.load_state_dict(torch.load('model_final.pth'))
model.eval()  # Set the model to evaluation mode

correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in eval_loader:  
        inputs = inputs.unsqueeze(1).squeeze(-1) 
        labels = torch.argmax(labels, dim=1) 

        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')
