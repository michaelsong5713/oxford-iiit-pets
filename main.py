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

def t_transforms_fn(image, mask):
    return train_transform(image), mask_transform(mask)

def v_transforms_fn(image, mask):
    return train_transform(image), mask_transform(mask)

train_data = datasets.OxfordIIITPet(
    root = "data",
    split = "trainval",
    target_types = "segmentation",
    transforms = t_transforms_fn,
    download = True
)

test_dataset = datasets.OxfordIIITPet(
    root = "data",
    split = "test",
    target_types = "segmentation",
    transforms = v_transforms_fn,
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

mask_transform = v2.Compose([
    v2.Resize((256, 256), interpolation=v2.InterpolationMode.NEAREST),
    v2.ToImage(),
    v2.Lambda(lambda x: x.squeeze(0).long() - 1)
])

'''
train_data.transform = train_transform
test_dataset.transform = test_transform
train_data.target_transform = mask_transform
test_dataset.target_transform = mask_transform
'''

training_len = len(train_data)
val_size = int(training_len*0.2)
train_size = training_len-val_size
train_dataset,val_dataset = random_split(
    train_data,
    [train_size,val_size]
)

train_dataloader = DataLoader(train_dataset,batch_size=64,shuffle=True)
val_dataloader = DataLoader(val_dataset,batch_size=64,shuffle=False)
test_dataloader = DataLoader(test_dataset,batch_size=64,shuffle=False)

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

class down(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2),
            double_conv_block(in_ch,out_ch)
        )
    def forward(self,x):
        return self.block(x)

class up(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.upscale = nn.ConvTranspose2d(in_ch,out_ch,kernel_size=2,stride=2)
        self.conv = double_conv_block(in_ch,out_ch)
    def forward(self,input,skip):
        input = self.upscale(input)
        return self.conv(torch.cat([input,skip],dim=1))

class neural_network(nn.Module):
    def __init__(self, in_ch = 3, features = 64, classes = 3):
        super().__init__()
        #start at 256x256
        self.encoder1 = double_conv_block(in_ch, features) #256x256 64 channels
        self.encoder2 = down(features, features*2) #128x128 128 channels
        self.encoder3 = down(features*2, features*4) #64x64 256 channels
        self.encoder4 = down(features*4, features*8) #32x32 512 channels

        self.bottleneck = down(features*8,features*16) #16x16 1024 channels

        self.decoder4 = up(features*16,features*8)
        self.decoder3 = up(features*8,features*4)
        self.decoder2 = up(features*4,features*2)
        self.decoder1 = up(features*2,features)

        self.output = nn.Conv2d(features,classes,kernel_size=1)
        #encoder

    def forward(self,x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        b1 = self.bottleneck(e4)

        d = self.decoder4(b1,e4)
        d = self.decoder3(d,e3)
        d = self.decoder2(d,e2)
        d = self.decoder1(d,e1)

        return self.output(d)
    def train_epoch(self,t_dataloader,v_dataloader,loss_fn,optimizer,scheduler,device):
        self.train()
        train_loss = 0
        num = 0
        for batch_d,batch_l in t_dataloader:
            batch_d, batch_l = batch_d.to(device), batch_l.to(device)
            #print(f"Mask Shape: {batch_l.shape}")
            output = self(batch_d)
            loss = loss_fn(output,batch_l)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss+=loss.item()
            num+=1
        train_avg_loss = train_loss/num
        print(f"Training Loss: {train_avg_loss}")
        self.eval()
        val_loss = 0
        num = 0
        with torch.no_grad():
            for batch_d,batch_l in v_dataloader:
                batch_d, batch_l = batch_d.to(device), batch_l.to(device)
                output = self(batch_d)
                loss = loss_fn(output,batch_l)
                val_loss+=loss.item()
                num+=1
        val_avg_loss = val_loss/num
        print(f"Validation Loss: {val_avg_loss}")
        scheduler.step()
    def test(self):

        pass

generations=10
model = neural_network().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
scheduler = CosineAnnealingLR(optimizer,T_max=generations)

for i in range(generations):
    print(f"Epoch: {i+1}\n")
    model.train_epoch(train_dataloader,val_dataloader,loss_fn,optimizer,scheduler,device)

