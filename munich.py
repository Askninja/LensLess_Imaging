import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.amp import autocast, GradScaler
import torch.nn.functional as F

# Residual Block similar to ResNet
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.bn_residual = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = self.residual_conv(x)
        residual = self.bn_residual(residual)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)

        x += residual  # Add residual connection
        x = self.relu(x)
        return x

# UNet++ with Residual Blocks (Hybrid of UNet++ and ResNet)
class UNetPlusPlus_ResNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNetPlusPlus_ResNet, self).__init__()
        # Use ResidualBlock instead of ConvBlock for deeper learning
        self.enc1 = ResidualBlock(in_channels, 64)
        self.enc2 = ResidualBlock(64, 128)
        self.enc3 = ResidualBlock(128, 256)
        self.enc4 = ResidualBlock(256, 512)
        self.bottleneck = ResidualBlock(512, 1024)

        self.dec4_1 = ResidualBlock(1024 + 512, 512)
        self.dec3_1 = ResidualBlock(512 + 256, 256)
        self.dec2_1 = ResidualBlock(256 + 128, 128)
        self.dec1_1 = ResidualBlock(128 + 64, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(F.max_pool2d(s1, 2))
        s3 = self.enc3(F.max_pool2d(s2, 2))
        s4 = self.enc4(F.max_pool2d(s3, 2))
        b = self.bottleneck(F.max_pool2d(s4, 2))

        d4 = self.dec4_1(torch.cat([F.interpolate(b, scale_factor=2), s4], dim=1))
        d3 = self.dec3_1(torch.cat([F.interpolate(d4, scale_factor=2), s3], dim=1))
        d2 = self.dec2_1(torch.cat([F.interpolate(d3, scale_factor=2), s2], dim=1))
        d1 = self.dec1_1(torch.cat([F.interpolate(d2, scale_factor=2), s1], dim=1))

        output = self.final_conv(d1)
        return output


# Dataset class with added noise and validation split
class ParticleDataset(Dataset):
    def __init__(self, input1_dir, gt_dir, transform=None, add_noise=False):
        self.input1_images = sorted([os.path.join(input1_dir, f) for f in os.listdir(input1_dir) if f.endswith('.png')])
        self.gt_images = sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.endswith('.png')])
        self.transform = transform
        self.add_noise = add_noise

    def __len__(self):
        return len(self.input1_images)

    def __getitem__(self, idx):
        input1_image = Image.open(self.input1_images[idx]).convert('L')
        gt_image = Image.open(self.gt_images[idx]).convert('L')

        if self.transform:
            input1_image = self.transform(input1_image)
            gt_image = self.transform(gt_image)

        if self.add_noise:
            noise = torch.randn_like(input1_image) * 0.1  # Adding noise
            input1_image = input1_image + noise

        return input1_image, gt_image

# Training function with tqdm for monitoring
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    scaler = GradScaler()  # Initialize the mixed precision scaler
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Use tqdm for batch progress in the training loop
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", file=sys.stdout) as pbar:
            for inputs, gts in train_loader:
                inputs, gts = inputs.to(device), gts.to(device)

                optimizer.zero_grad()

                
                with autocast(device_type='cuda'):  
                    outputs = model(inputs)
                    loss = criterion(outputs, gts)

                scaler.scale(loss).backward()  # Scale the loss for mixed precision
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                pbar.set_postfix({"Loss": running_loss / len(train_loader)})
                pbar.update(1)

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, gts in val_loader:
                inputs, gts = inputs.to(device), gts.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, gts)
                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}')

        # Save the model after each epoch
        torch.save(model.state_dict(), f'/home/abhiram001/part2_resnet_unetplusplus_epoch_{epoch+1}.pth')
        print(f'Model saved after Epoch {epoch+1}')

    # Plot loss curve
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss vs Epochs')
    plt.legend()
    plt.savefig('/home/abhiram001/part2_resnet_unetplusplus_training_loss_curve.png')
    plt.show()

    return model

# Main function for training
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    input1_dir = '/home/abhiram001/train/Input'
    gt_dir = '/home/abhiram001/GT2/GT'

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    dataset = ParticleDataset(input1_dir=input1_dir, gt_dir=gt_dir, transform=transform, add_noise=True)
    
    # Split dataset into 80% train and 20% validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

    model = UNetPlusPlus_ResNet(in_channels=1, out_channels=1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)  # L2 regularization via weight_decay

    # Train model and plot loss curves
    num_epochs = 50
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

 
    torch.save(trained_model.state_dict(), '/home/abhiram001/part2_resnet_unetplusplus_final_model.pth')

if __name__ == '__main__':
    main()
