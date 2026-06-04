import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR

device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0),"\n")

train_data = datasets.OxfordIIITPet(
    root = "data",
    split = "trainval",
    target_types = "segmentation",
    download = True
)

test_dataset = datasets.OxfordIIITPet(
    root = "data",
    split = "test",
    target_types = "segmentation",
    download = True
)

'''
fig, axes = plt.subplots(5, 2, figsize=(6, 30))
for i in range(5):
    image,mask = test_dataset[i]
    plt.imshow(image)
    image_array = np.array(image)
    mask_array = np.array(mask)
    print(f"Image shape:{image_array.shape}")
    print(f"Mask shape:{mask_array.shape}")

    axes[i, 0].imshow(image_array)
    axes[i, 0].set_title(f"Image {i}")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(mask_array)
    axes[i, 1].set_title(f"Mask {i} | unique values: {np.unique(mask_array)}")
    axes[i, 1].axis("off")

plt.tight_layout()
plt.show()
'''

mean =  ([0.485, 0.456, 0.406])
std = ([0.229, 0.224, 0.225])

train_transform = v2.Compose([
    v2.Resize((256, 256)),
    v2.ToImage(),
    v2.ToDtype(torch.float32,scale=True),
    v2.Normalize(mean=mean,std=std)
])

test_transform = v2.Compose([
    v2.Resize((256, 256)),
    v2.ToImage(),
    v2.ToDtype(torch.float32,scale=True),
    v2.Normalize(mean=mean,std=std)
])

train_data.transform = train_transform
test_dataset.transform = test_transform

training_len = len(train_data)
val_size = int(training_len*0.2)
train_size = training_len-val_size
train_dataset,val_dataset = random_split(
    train_data,
    [train_size,val_size]
)

train_dataloader = DataLoader(train_dataset,batch_size=64,shuffle=True)
val_dataloader = DataLoader(val_dataset,batch_size=64,shuffle=False)
train_dataloader = DataLoader(test_dataset,batch_size=64,shuffle=False)

class double_conv_block(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch,out_ch,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
    def forward(self,x):
        return self.block(x)


temp = double_conv_block(3,64)
test = torch.randn(1,3,256,256)
out = temp(test)
print(out.shape)

class neural_network(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        pass
    def train(self):
        pass
    def test(self):
        pass