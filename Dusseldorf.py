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
import torch.nn.functional as F
import cv2  # For image processing and particle counting

# Custom Combined Dice and BCE Loss
class DiceBCEWithLogitsLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCEWithLogitsLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(weight=weight, reduction='mean')

    def forward(self, inputs, targets, smooth=1):
        # BCE Loss
        bce_loss = self.bce(inputs, targets)

        # Dice Loss
        inputs = torch.sigmoid(inputs)  # Apply sigmoid to get probabilities
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        # Combine BCE and Dice Loss
        return bce_loss + dice_loss

# Residual Block with Dropout
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout1 = nn.Dropout2d(p=dropout_prob)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.bn_residual = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout2d(p=dropout_prob)

    def forward(self, x):
        residual = self.residual_conv(x)
        residual = self.bn_residual(residual)
        residual = self.dropout2(residual)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual  # Add residual connection
        out = self.relu(out)
        return out

# UNet++ with Residual Blocks and Dropout
class UNetPlusPlus_ResNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, dropout_prob=0.3):
        super(UNetPlusPlus_ResNet, self).__init__()
        # Encoder
        self.enc1 = ResidualBlock(in_channels, 64, dropout_prob=dropout_prob)
        self.enc2 = ResidualBlock(64, 128, dropout_prob=dropout_prob)
        self.enc3 = ResidualBlock(128, 256, dropout_prob=dropout_prob)
        self.enc4 = ResidualBlock(256, 512, dropout_prob=dropout_prob)
        self.bottleneck = ResidualBlock(512, 1024, dropout_prob=dropout_prob)

        # Decoder
        self.dec4_1 = ResidualBlock(1024 + 512, 512, dropout_prob=dropout_prob)
        self.dec3_1 = ResidualBlock(512 + 256, 256, dropout_prob=dropout_prob)
        self.dec2_1 = ResidualBlock(256 + 128, 128, dropout_prob=dropout_prob)
        self.dec1_1 = ResidualBlock(128 + 64, 64, dropout_prob=dropout_prob)

        # Final Convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        s1 = self.enc1(x)  # [B, 64, H, W]
        s2 = self.enc2(F.max_pool2d(s1, 2))  # [B, 128, H/2, W/2]
        s3 = self.enc3(F.max_pool2d(s2, 2))  # [B, 256, H/4, W/4]
        s4 = self.enc4(F.max_pool2d(s3, 2))  # [B, 512, H/8, W/8]
        b = self.bottleneck(F.max_pool2d(s4, 2))  # [B, 1024, H/16, W/16]

        # Decoder with Skip Connections
        d4 = self.dec4_1(torch.cat([F.interpolate(b, scale_factor=2, mode='bilinear', align_corners=True), s4], dim=1))
        d3 = self.dec3_1(torch.cat([F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=True), s3], dim=1))
        d2 = self.dec2_1(torch.cat([F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True), s2], dim=1))
        d1 = self.dec1_1(torch.cat([F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True), s1], dim=1))

        # Output Layer
        output = self.final_conv(d1)
        return output

# Dataset class with added Gaussian noise (2-5%) and validation split
class ParticleDataset(Dataset):
    def __init__(self, input1_dir, gt_dir, transform=None, add_noise=False, noise_range=(0.02, 0.05)):
        """
        Args:
            input1_dir (str): Directory with input images.
            gt_dir (str): Directory with ground truth images.
            transform (callable, optional): Optional transform to be applied on a sample.
            add_noise (bool): Whether to add Gaussian noise.
            noise_range (tuple): Range of noise levels to add (min, max).
        """
        self.input1_images = sorted([os.path.join(input1_dir, f) for f in os.listdir(input1_dir) if f.endswith('.png')])
        self.gt_images = sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.endswith('.png')])
        self.transform = transform
        self.add_noise = add_noise
        self.noise_min = noise_range[0]
        self.noise_max = noise_range[1]

    def __len__(self):
        return len(self.input1_images)

    def __getitem__(self, idx):
        input1_image = Image.open(self.input1_images[idx]).convert('L')
        gt_image = Image.open(self.gt_images[idx]).convert('L')

        if self.transform:
            input1_image = self.transform(input1_image)
            gt_image = self.transform(gt_image)

        if self.add_noise:
            noise_level = np.random.uniform(self.noise_min, self.noise_max)
            noise = torch.randn_like(input1_image) * noise_level  # Adding Gaussian noise
            input1_image = input1_image + noise
            input1_image = torch.clamp(input1_image, 0., 1.)  # Ensure pixel values are within [0,1]

        return input1_image, gt_image

# Function to count particles using connected components
def count_particles(pred_mask, threshold=0.5, min_size=5):
    """
    Counts the number of particles in the predicted mask.

    Args:
        pred_mask (torch.Tensor): Predicted mask tensor with shape [1, H, W].
        threshold (float): Threshold to binarize the mask.
        min_size (int): Minimum size of particles to be counted.

    Returns:
        int: Number of particles detected.
    """
    # Move tensor to CPU and convert to numpy
    pred_mask = pred_mask.squeeze().cpu().numpy()

    # Apply sigmoid and threshold
    pred_mask = 1 / (1 + np.exp(-pred_mask))  # Sigmoid
    pred_mask = (pred_mask > threshold).astype(np.uint8)

    # Remove small objects
    num_labels, labels = cv2.connectedComponents(pred_mask, connectivity=8)

    # Optionally filter out small particles
    count = 0
    for label in range(1, num_labels):  # Start from 1 to exclude background
        component = (labels == label).astype(np.uint8)
        if cv2.countNonZero(component) >= min_size:
            count += 1

    return count
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Please check your GPU setup.")
    
    # Set the device (mapped GPUs are cuda:0 and cuda:1)
    device = torch.device(device)  # Typically 'cuda:0'
    model.to(device)

    # Wrap the model for multi-GPU training using DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training: {device}")
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())), output_device=0)
    
    train_losses, val_losses = [], []
    train_counts, val_counts = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_count = 0
        total_batches = 0

        # Use tqdm for batch progress in the training loop
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", file=sys.stdout) as pbar:
            for inputs, gts in train_loader:
                inputs, gts = inputs.to(device), gts.to(device)

                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, gts)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                total_batches += 1

                # Optional: Compute particle counts for training data
                # Assuming batch size of 1 for accurate counting; modify if batch size >1
                for i in range(outputs.size(0)):
                    count = count_particles(outputs[i].detach())
                    running_count += count

                pbar.set_postfix({
                    "Loss": running_loss / total_batches,
                    "Count": running_count / total_batches
                })
                pbar.update(1)

        epoch_loss = running_loss / len(train_loader)
        epoch_count = running_count / len(train_loader)
        train_losses.append(epoch_loss)
        train_counts.append(epoch_count)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_count = 0.0
        with torch.no_grad():
            for inputs, gts in val_loader:
                inputs, gts = inputs.to(device), gts.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, gts)
                val_loss += loss.item()

                # Compute particle counts for validation data
                for i in range(outputs.size(0)):
                    count = count_particles(outputs[i])
                    val_count += count

        val_loss = val_loss / len(val_loader)
        val_count = val_count / len(val_loader)
        val_losses.append(val_loss)
        val_counts.append(val_count)

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {epoch_loss:.4f}, Train Count: {epoch_count:.2f}, '
              f'Validation Loss: {val_loss:.4f}, Validation Count: {val_count:.2f}')

        # Step the scheduler based on validation loss
        scheduler.step(val_loss)

        # Save the model after each epoch
        save_path = f'/usr1/home/s124mdg42_02/ureca/Dusseldorf_epoch{epoch+1}.pth'
        torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), save_path)
        print(f'Model saved after Epoch {epoch+1} at {save_path}')

    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss vs Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('/usr1/home/s124mdg42_02/ureca/Dusseldorf_Loss.png')
    plt.show()

    # Plot particle counts
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_counts, label='Train Avg Count')
    plt.plot(range(1, num_epochs + 1), val_counts, label='Validation Avg Count')
    plt.xlabel('Epochs')
    plt.ylabel('Average Particle Count')
    plt.title('Train and Validation Average Particle Count vs Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('/usr1/home/s124mdg42_02/ureca/Dusseldorf_Particle.png')
    plt.show()

    return model
# Main function for training
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    input1_dir = '/usr1/home/s124mdg42_02/ureca/Input'  # Directory with input images (iSCAT measurements)
    gt_dir = '/usr1/home/s124mdg42_02/ureca/GT'           # Directory with ground truth images

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    dataset = ParticleDataset(input1_dir=input1_dir, gt_dir=gt_dir, transform=transform, add_noise=True, noise_range=(0.02, 0.05))
    
    # Split dataset into 80% train and 20% validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=20, shuffle=False, num_workers=4, pin_memory=True)

    model = UNetPlusPlus_ResNet(in_channels=1, out_channels=1, dropout_prob=0.3)
    criterion = DiceBCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)  # L2 regularization via weight_decay

    # Learning rate scheduler: Reduce LR by factor of 0.5 if validation loss doesn't improve for 5 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Train model and plot loss curves
    num_epochs = 50
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device)

    # Save the final trained model
    if torch.cuda.device_count() > 1:
        torch.save(trained_model.module.state_dict(), '/usr1/home/s124mdg42_02/ureca/Dusseldorf.pth')
    else:
        torch.save(trained_model.state_dict(), '/usr1/home/s124mdg42_02/ureca/Dusseldorf.pth')

if __name__ == '__main__':
    main()
