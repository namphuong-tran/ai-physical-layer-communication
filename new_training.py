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
folder_name = "benchmark/" + current_time.strftime("%Y%m%d_%H%M%S")
os.makedirs(folder_name, exist_ok=True)
output_dir = 'output_data/'

def load_files(prefix):
    y_files = sorted(glob.glob(output_dir + f'{prefix}/{prefix}_data_*.npy'))
    t_files = sorted(glob.glob(output_dir + f'{prefix}/{prefix}_labels_*.npy'))
    return y_files, t_files

train_y_files, train_t_files = load_files('train')
eval_y_files, eval_t_files = load_files('test')

train_dataset = ChunkedTimingDataset(train_y_files, train_t_files)
eval_dataset = ChunkedTimingDataset(eval_y_files, eval_t_files)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

input_length = train_dataset.y_data.shape[1]
num_classes = train_dataset.t_data.shape[1]

model = CNNSynchronizer(input_length, num_classes).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=True)

train_losses = []
best_loss = float('inf')
num_epochs = int(1e5)
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for inputs, labels in train_loader:
        inputs = inputs.unsqueeze(1).squeeze(-1).cuda()
        labels = torch.argmax(labels, dim=1).cuda()
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_epoch_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}')
    
    if (epoch + 1) % 100 == 0:
        with open(folder_name + '/training_loss.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if (epoch + 1) == 100:
                writer.writerow(['Epoch', 'Loss'])
            for e, loss in enumerate(train_losses[-100:]):
                writer.writerow([epoch + 1 - 99 + e, loss])

    if (epoch + 1) % 20 == 0:
        torch.save(model.state_dict(), folder_name + f'/model_epoch_{epoch+1}.pth')
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), folder_name + '/best_model.pth')
            print(f'Model saved at epoch {epoch+1} with loss {avg_epoch_loss:.4f}')
    
    scheduler.step(avg_epoch_loss)

if num_epochs % 100 != 0:
    remaining_epochs = num_epochs % 100
    with open(folder_name + '/training_loss.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if num_epochs < 100:
            writer.writerow(['Epoch', 'Loss'])
        for e, loss in enumerate(train_losses[-remaining_epochs:]):
            writer.writerow([num_epochs - remaining_epochs + e + 1, loss])

torch.save(model.state_dict(), folder_name + '/model_final.pth')