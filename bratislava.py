import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.amp import autocast, GradScaler
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNetPlusPlus, self).__init__()
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.bottleneck = ConvBlock(512, 1024)

        self.dec4_1 = ConvBlock(1024 + 512, 512)
        self.dec3_1 = ConvBlock(512 + 256, 256)
        self.dec2_1 = ConvBlock(256 + 128, 128)
        self.dec1_1 = ConvBlock(128 + 64, 64)

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


# PatchGAN Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels=2):  # input and target images are concatenated along the channel dimension
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)


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


# Loss function for GAN
def adversarial_loss(predicted, target_is_real):
    if target_is_real:
        target_tensor = torch.ones_like(predicted)
    else:
        target_tensor = torch.zeros_like(predicted)
    return F.mse_loss(predicted, target_tensor)

# Training the GAN with U-Net Generator
def train_gan(generator, discriminator, train_loader, val_loader, num_epochs, device):
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=1e-4)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4)

    # Loss functions
    adversarial_criterion = adversarial_loss  # Adversarial loss (MSE)
    l1_loss = nn.L1Loss()  # Pixel-wise loss (L1)

    generator.to(device)
    discriminator.to(device)

    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        g_running_loss = 0.0
        d_running_loss = 0.0

        # Training loop
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", file=sys.stdout) as pbar:
            for inputs, gts in train_loader:
                inputs, gts = inputs.to(device), gts.to(device)

                # Train Discriminator
                optimizer_D.zero_grad()
                # Generate fake images
                fake_outputs = generator(inputs)

                # Concatenate real and fake images for the discriminator
                real_input = torch.cat((inputs, gts), 1)
                fake_input = torch.cat((inputs, fake_outputs.detach()), 1)

                # Real loss
                real_loss = adversarial_criterion(discriminator(real_input), True)
                # Fake loss
                fake_loss = adversarial_criterion(discriminator(fake_input), False)

                # Total loss for discriminator
                d_loss = (real_loss + fake_loss) * 0.5
                d_loss.backward()
                optimizer_D.step()

                # Train Generator
                optimizer_G.zero_grad()

                # Adversarial loss
                fake_input = torch.cat((inputs, fake_outputs), 1)
                g_adv_loss = adversarial_criterion(discriminator(fake_input), True)

                # L1 loss (reconstruction loss)
                g_l1_loss = l1_loss(fake_outputs, gts)

                # Total generator loss
                g_loss = g_adv_loss + 100 * g_l1_loss  # Lambda = 100 for pixel loss
                g_loss.backward()
                optimizer_G.step()

                g_running_loss += g_loss.item()
                d_running_loss += d_loss.item()

                pbar.set_postfix({"G Loss": g_running_loss / len(train_loader), "D Loss": d_running_loss / len(train_loader)})
                pbar.update(1)

        

        print(f'Epoch [{epoch+1}/{num_epochs}], G Loss: {g_running_loss:.4f}, D Loss: {d_running_loss:.4f}')

    return generator, discriminator


# Main function for GAN training
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

    # Initialize Generator (U-Net++) and Discriminator (PatchGAN)
    generator = UNetPlusPlus(in_channels=1, out_channels=1)
    discriminator = Discriminator(in_channels=2)  # Concatenated input and output channels

    # Train GAN
    num_epochs = 50
    trained_generator, trained_discriminator = train_gan(generator, discriminator, train_loader, val_loader, num_epochs, device)

    # Save the final model
    torch.save(trained_generator.state_dict(), '/home/abhiram001/part2_unetplusplus_gan_generator.pth')
    torch.save(trained_discriminator.state_dict(), '/home/abhiram001/part2_unetplusplus_gan_discriminator.pth')

if __name__ == '__main__':
    main()
