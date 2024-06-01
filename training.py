import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import csv
from dataset import ChunkedTimingDataset
import glob
from synchronizer import CNNSynchronizer
# Function to load files
def load_files(prefix):
    y_files = sorted(glob.glob(f'{prefix}/{prefix}_y_*.npy'))
    t_files = sorted(glob.glob(f'{prefix}/{prefix}_t_*.npy'))
    return y_files, t_files

# Load the training and evaluation datasets from files
train_y_files, train_t_files = load_files('train')
eval_y_files, eval_t_files = load_files('eval')

# Create the datasets and DataLoaders
train_dataset = ChunkedTimingDataset(train_y_files, train_t_files)
eval_dataset = ChunkedTimingDataset(eval_y_files, eval_t_files)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)


input_length = train_dataset.y_data.shape[1]  
num_classes = train_dataset.t_data.shape[1]   

# Initialize the model, define the loss function and the optimizer
model = CNNSynchronizer(input_length, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize lists to store loss values
train_losses = []

num_epochs = 10
for epoch in range(num_epochs):
    epoch_loss = 0
    for inputs, labels in train_loader:
        # Ensure the inputs are of shape (batch_size, 1, input_length)
        inputs = inputs.unsqueeze(1).squeeze(-1)

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
        epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_epoch_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}')

    # Save the training loss every 500 or 1000 epochs
    if (epoch + 1) % 500 == 0:
        filename = f'training_loss_epoch_{epoch+1}.csv'
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Epoch', 'Loss'])
            for e, loss in enumerate(train_losses):
                writer.writerow([e + 1, loss])
    if (epoch + 1) % 1000 == 0:
        # Save the model checkpoint
        torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')

# Final save of all epochs
with open('training_loss_final.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Epoch', 'Loss'])
    for epoch, loss in enumerate(train_losses):
        writer.writerow([epoch + 1, loss])

# Save the final model
torch.save(model.state_dict(), 'model_final.pth')