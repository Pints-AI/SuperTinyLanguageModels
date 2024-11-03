import numpy as np
import torch
from torch.utils.data import Dataset
import random

class EndByteClassifierDataset(Dataset):
    """
    Dataset to load data from binary files for training with a fixed context window.
    """
    def __init__(self, x_path: str, y_path: str, config: dict):
        """
        x_path: Path to the `train_x.bin` or `val_x.bin` file.
        y_path: Path to the `train_y.bin` or `val_y.bin` file.
        config: contains details about the dataset
        """
        self.x_data = np.memmap(x_path, dtype=np.uint8, mode="r")
        self.y_data = np.memmap(y_path, dtype=np.uint8, mode="r")
        self.context_window = config['model']['context_window']
        self.device = config["general"]["device"]

        # need to double check with calvin if this is intended!
        self.dataset_len = (len(self.x_data) - self.context_window + 1) // self.context_window
        assert len(self.x_data) == len(self.y_data), "Input and label files must have the same length."

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        start_idx = idx * self.context_window
        x = torch.tensor(self.x_data[start_idx : start_idx + self.context_window], dtype=torch.long, device=self.device)
        y = torch.tensor(self.y_data[start_idx : start_idx + self.context_window], dtype=torch.float32, device=self.device)
        return x, y
