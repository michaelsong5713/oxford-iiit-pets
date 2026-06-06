import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import JaccardIndex
from dice_loss import DiceLoss

device = "cuda" if torch.cuda.is_available() else "cpu"
if __name__ == '__main__':
    if torch.cuda.is_available():
        print("GPU Name:", torch.cuda.get_device_name(0),"\n")

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

def t_transforms_fn(image, mask):
    return train_transform(image), mask_transform(mask)

def v_transforms_fn(image, mask):
    return test_transform(image), mask_transform(mask)


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
    def visualize(self,viz_images,viz_masks,device,num_images,epoch):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        self.eval()
        images,masks = viz_images.to(device),viz_masks.to(device)
        #print(f"{images.shape}")
        #print(f"{masks.shape}")
        with torch.no_grad():
            output = self(images).argmax(dim=1) #16x256x256
        fig, axes = plt.subplots(num_images, 3, figsize=(10, num_images * 3))
        axes[0, 0].set_title("Image")
        axes[0, 1].set_title("True Mask")
        axes[0, 2].set_title("Predicted Mask")
        pil_fn = v2.ToPILImage()
        for i in range(int(num_images)):
            cur_image,cur_mask = images[i],masks[i]
            cur_output = output[i]
            cur_image = cur_image.cpu()*std+mean
            cur_image = pil_fn(cur_image)
            axes[i, 0].imshow(cur_image)
            axes[i, 1].imshow(cur_mask.cpu().numpy(), cmap='tab10', vmin=0, vmax=2)
            axes[i, 2].imshow(cur_output.cpu().numpy(), cmap='tab10', vmin=0, vmax=2)
            for ax in axes[i]: ax.axis("off")
        plt.suptitle(f"Epoch {epoch}")
        plt.tight_layout()
        plt.savefig(f"outputs/epoch_{epoch}.png")
        plt.close()

    def train_epoch(self,t_dataloader,v_dataloader,loss_fn,optimizer,scheduler,device,epoch,iou_fn,dice_loss_fn,viz_images,viz_masks):
        self.train()
        train_loss = 0
        num = 0
        for batch_d,batch_l in t_dataloader:
            batch_d,batch_l = batch_d.to(device),batch_l.to(device)
            optimizer.zero_grad()
            #print(f"Mask Shape: {batch_l.shape}")
            output = self(batch_d)
            loss = loss_fn(output,batch_l)+dice_loss_fn(output,batch_l)
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
            num+=1
            #train_avg_loss = train_loss/num
            #print(f"Training Loss: {train_avg_loss}")
            iou_fn.update(output.argmax(dim=1),batch_l)
        train_avg_loss = train_loss/num
        print(f"Training Loss: {train_avg_loss}")
        train_iou = iou_fn.compute()
        iou_fn.reset()
        print(f"Training IoU: {train_iou}")
        self.eval()
        val_loss = 0
        num = 0
        with torch.no_grad():
            for batch_d,batch_l in v_dataloader:
                batch_d,batch_l = batch_d.to(device),batch_l.to(device)
                output = self(batch_d)
                loss = loss_fn(output,batch_l)+dice_loss_fn(output,batch_l)
                val_loss+=loss.item()
                num+=1
                iou_fn.update(output.argmax(dim=1),batch_l)
        scheduler.step()
        self.visualize(viz_images,viz_masks,device,5,epoch)
        val_avg_loss = val_loss/num
        print(f"Validation Loss: {val_avg_loss}")
        val_iou = iou_fn.compute()
        iou_fn.reset()
        print(f"Validation IoU: {val_iou}")
        return val_iou

    def test(self,test_dataloader,device,iou_fn):
        with torch.no_grad():
            for batch_d,batch_l in test_dataloader:
                batch_d,batch_l = batch_d.to(device),batch_l.to(device)
                output = self(batch_d)
                iou_fn.update(output.argmax(dim=1),batch_l)
        test_iou = iou_fn.compute()
        print(f"Final IOU: {test_iou}")
        pass

if __name__ == '__main__':
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

    training_len = len(train_data)
    val_size = int(training_len*0.2)
    train_size = training_len-val_size
    train_dataset,val_dataset = random_split(
        train_data,
        [train_size,val_size]
    )
    train_dataloader = DataLoader(train_dataset,batch_size=16,shuffle=True,num_workers=4,pin_memory=True,persistent_workers=True)
    val_dataloader = DataLoader(val_dataset,batch_size=16,shuffle=False,num_workers=4,pin_memory=True,persistent_workers=True)
    test_dataloader = DataLoader(test_dataset,batch_size=16,shuffle=False,num_workers=4,pin_memory=True,persistent_workers=True)

    generations=40
    model = neural_network().to(device)

    counts = torch.tensor([0,0,0])
    for labels,masks in train_dataloader:
        for i in range(3):
            counts[i]+=(masks==i).sum()
    total = counts.sum()
    weights =  total/(3*counts)
    weights = weights.to(device)

    dice_loss_fn = DiceLoss()
    loss_fn = nn.CrossEntropyLoss(weight = weights)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    scheduler = CosineAnnealingLR(optimizer,T_max=generations)
    iou_fn = JaccardIndex(task="multiclass", num_classes=3).to(device)
    viz_images, viz_masks = next(iter(val_dataloader))

    best_iou = 0
    for i in range(generations):
        print(f"Epoch: {i+1}\n")
        val_iou = model.train_epoch(train_dataloader,val_dataloader,loss_fn,optimizer,scheduler,device,i+1,iou_fn,dice_loss_fn,viz_images,viz_masks)
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), "models/best_model.pth")
    print(f"\nTraining Done\n")
    model.test(test_dataloader,device,iou_fn)
    torch.save(model.state_dict(), "models/final_model.pth")