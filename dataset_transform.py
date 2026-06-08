import torch
from torch.utils.data import Dataset

class Dataset_Creator(Dataset):
        def __init__(self,subset,transform):
            self.subset = subset
            self.transform_fn = transform
        def __len__(self):
            return len(self.subset)
        def __getitem__(self,idx):
            image,mask = self.subset[idx]
            return self.transform_fn(image,mask)