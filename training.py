import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import csv
from dataset import ChunkedTimingDataset
import glob
from synchronizer import CNNSynchronizer
import datetime
import os
current_time = datetime.datetime.now()
folder_name = "benchmark/" + \
    current_time.strftime("%Y%m%d_%H%M%S")
os.makedirs(folder_name, exist_ok=True)
output_dir = 'output_data/'

# Function to load files
def load_files(prefix):
    y_files = sorted(glob.glob(output_dir + f'{prefix}/{prefix}_data_*.npy'))
    t_files = sorted(glob.glob(output_dir + f'{prefix}/{prefix}_labels_*.npy'))
    return y_files, t_files

# Load the training and evaluation datasets from files
train_y_files, train_t_files = load_files('train')
eval_y_files, eval_t_files = load_files('test')

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
optimizer = optim.Adam(model.parameters(), lr=0.0005)
# StepLR scheduler: reduce the learning rate by a factor of 0.1 every 1000 epochs
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
# Scheduler initialization with ReduceLROnPlateau
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=True)


# Initialize lists to store loss values
train_losses = []
best_loss = float('inf')
num_epochs = int(1e5)
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

    # Save the training loss every 1000 epochs
    if (epoch + 1) % 1000 == 0:
        with open(folder_name + '/training_loss.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if (epoch + 1) == 1000:
                writer.writerow(['Epoch', 'Loss'])  # Write header only once
            for e, loss in enumerate(train_losses[-1000:]):
                writer.writerow([epoch + 1 - 999 + e, loss])

    # Save the model checkpoint
    if (epoch + 1) % 2000 == 0:
        torch.save(model.state_dict(), folder_name + f'/model_epoch_{epoch+1}.pth')
    # Save the model every 100 epochs if the loss is lower than the best loss
    if (epoch + 1) % 100 == 0:
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), folder_name + '/best_model.pth')
            print(f'Model saved at epoch {epoch+1} with loss {avg_epoch_loss:.4f}')
    # Step the scheduler
    # scheduler.step() # StepLR
    scheduler.step(avg_epoch_loss) # ReduceLROnPlateau


# Final save of remaining epochs if not a multiple of 500
if num_epochs % 1000 != 0:
    remaining_epochs = num_epochs % 1000
    with open(folder_name + '/training_loss.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if num_epochs < 1000:
            writer.writerow(['Epoch', 'Loss'])  # Write header if it hasn't been written
        for e, loss in enumerate(train_losses[-remaining_epochs:]):
            writer.writerow([num_epochs - remaining_epochs + e + 1, loss])

# Save the final model
torch.save(model.state_dict(), folder_name + '/model_final.pth')