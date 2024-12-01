import torch
from torch.utils.data import Dataset

class SuperDARN1995(Dataset):
    def __init__(self, data_paths):
        self.data_files = data_paths
        self.data = []
        self.load_data(data_paths)

