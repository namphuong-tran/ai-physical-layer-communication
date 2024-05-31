import numpy as np
import torch
from torch.utils.data import Dataset

class ChunkedTimingDataset(Dataset):
    def __init__(self, y_files, t_files):
        self.y_files = y_files
        self.t_files = t_files
        self.y_data = []
        self.t_data = []
        self.load_data()

    def load_data(self):
        for y_file, t_file in zip(self.y_files, self.t_files):
            y_chunk = np.load(y_file)
            t_chunk = np.load(t_file)
            self.y_data.append(y_chunk)
            self.t_data.append(t_chunk)
        
        self.y_data = np.concatenate(self.y_data)
        self.t_data = np.concatenate(self.t_data)
        
    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        y_sample = self.y_data[idx]
        t_sample = self.t_data[idx]
        return torch.tensor(y_sample, dtype=torch.float32), torch.tensor(t_sample, dtype=torch.float32)
