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
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

# Ensure to set visible GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

# Residual Block similar to ResNet with added dropout for regularization
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)  # Adding dropout with a rate of 0.3
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.bn_residual = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = self.residual_conv(x)
        residual = self.bn_residual(residual)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)  # Applying dropout after the activation
        
        x = self.conv2(x)
        x = self.bn2(x)

        x += residual  # Add residual connection
        x = self.relu(x)
        return x

# UNet++ with Residual Blocks as Generator
class UNetPlusPlus_ResNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNetPlusPlus_ResNet, self).__init__()
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

# Early stopping class based on validation loss
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

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

# Training function for GAN with validation stopping and dropout layers
def train_gan(generator, discriminator, train_loader, val_loader, num_epochs, device):
    optimizer_G = optim.Adam(generator.parameters(), lr=1e-4, weight_decay=1e-5)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4, weight_decay=1e-5)

    adversarial_criterion = adversarial_loss  # Adversarial loss (MSE)
    l1_loss = nn.L1Loss()  # L1 loss (for image reconstruction)

    generator.to(device)
    discriminator.to(device)

    # For multi-GPU use
    generator = nn.DataParallel(generator)
    discriminator = nn.DataParallel(discriminator)

    scaler = GradScaler()  # Initialize the mixed precision scaler
    train_losses, val_losses = [], []
    early_stopping = EarlyStopping(patience=5, min_delta=0.01)

    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        g_running_loss = 0.0
        d_running_loss = 0.0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", file=sys.stdout) as pbar:
            for inputs, gts in train_loader:
                inputs, gts = inputs.to(device), gts.to(device)

                # Train Discriminator
                optimizer_D.zero_grad()
                fake_outputs = generator(inputs)

                real_input = torch.cat((inputs, gts), 1)
                fake_input = torch.cat((inputs, fake_outputs.detach()), 1)

                real_loss = adversarial_criterion(discriminator(real_input), True)
                fake_loss = adversarial_criterion(discriminator(fake_input), False)
                d_loss = (real_loss + fake_loss) * 0.5

                scaler.scale(d_loss).backward()
                scaler.step(optimizer_D)
                scaler.update()

                # Train Generator
                optimizer_G.zero_grad()
                fake_input = torch.cat((inputs, fake_outputs), 1)
                g_adv_loss = adversarial_criterion(discriminator(fake_input), True)
                g_l1_loss = l1_loss(fake_outputs, gts)
                g_loss = g_adv_loss + 100 * g_l1_loss  # Lambda = 100 for pixel loss

                scaler.scale(g_loss).backward()
                scaler.step(optimizer_G)
                scaler.update()

                g_running_loss += g_loss.item()
                d_running_loss += d_loss.item()

                pbar.set_postfix({"G Loss": g_running_loss / len(train_loader), "D Loss": d_running_loss / len(train_loader)})
                pbar.update(1)

        epoch_loss = g_running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # Validation phase
        generator.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, gts in val_loader:
                inputs, gts = inputs.to(device), gts.to(device)
                fake_outputs = generator(inputs)
                g_l1_loss = l1_loss(fake_outputs, gts)
                val_loss += g_l1_loss.item()

        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], G Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}')

        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break

    # Plot loss curve
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss vs Epochs')
    plt.legend()
    plt.savefig('/usr1/home/s124mdg42_02/ureca/New_GAN.png')
    plt.show()

    return generator, discriminator

# Main function for GAN training
def main():
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Print the GPUs being used
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"Visible GPUs: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    input1_dir = '/usr1/home/s124mdg42_02/ureca/Input'
    gt_dir = '/usr1/home/s124mdg42_02/ureca/GT'

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    dataset = ParticleDataset(input1_dir=input1_dir, gt_dir=gt_dir, transform=transform, add_noise=True)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=25, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=25, shuffle=False, num_workers=4)

    generator = UNetPlusPlus_ResNet(in_channels=1, out_channels=1)
    discriminator = Discriminator(in_channels=2)

    num_epochs = 50
    trained_generator, trained_discriminator = train_gan(generator, discriminator, train_loader, val_loader, num_epochs, device)

    torch.save(trained_generator.state_dict(), '/usr1/home/s124mdg42_02/ureca/generator')
    torch.save(trained_discriminator.state_dict(), '/usr1/home/s124mdg42_02/ureca/discriminator')

if __name__ == '__main__':
    main()
