import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR

train_data = datasets.OxfordIIITPet(
    root = "data",
    train = True,
    download = True
)

test_dataset = datasets.OxfordIIITPet(
    root = "data",
    train = False,
    download = True
)
mean =  ([0.485, 0.456, 0.406])
std = ([0.229, 0.224, 0.225])

train_transform = v2.Compose(
    v2.ToImage(),
    v2.ToDtype(torch.float32,scale=True),
    v2.Normalize(mean=mean,std=std)
)

test_transform = v2.Compose(
    v2.ToImage(),
    v2.ToDtype(torch.float32,scale=True),
    v2.Normalize(mean=mean,std=std)
)

training_len = len(train_data)
val_size = training_len*0.2
train_size = training_len-val_size
train_dataset,val_dataset = random_split(
    train_data,
    [train_size,val_size]
)
train_dataset.transform = train_transform
test_dataset.transform = test_transform
val_dataset.transform = test_transform

def conv_block(nn.Module):
    def __init__(self,out_ch,in_ch):
        super().__init__()
    def forward(self,x):
        pass

def neural_network(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        pass
    def train(self):
        pass
    def test(self):
        pass