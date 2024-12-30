"""
FNet with Wavelet Transform and Fourier Transforms.
MSE Loss + BCE Loss + Perceptual Loss
"""
import glob
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
import numpy as np
import scipy
import pywt
import matplotlib.pyplot as plt

from PIL import Image
import skimage
from torchvision import transforms
import torchvision

def normalize(x):
    max_value = x.max()
    min_value = x.min()
    return (x - min_value) / (max_value - min_value)

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.input_images = sorted(os.listdir(input_dir))
        self.input_images = [f for f in self.input_images if f.endswith('.png')]
        self.target_images = sorted(os.listdir(target_dir))
        self.target_images = [f for f in self.target_images if f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        # Load input and target images
        input_image_path = os.path.join(self.input_dir, self.input_images[idx])
        target_image_path = os.path.join(self.target_dir, self.target_images[idx])
        
        input_image = Image.open(input_image_path)
        target_image = Image.open(target_image_path)
        
        # Apply transformations
        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)
            
        return input_image, target_image
      
class FirstOrderWaveletTransform(nn.Module):
    def __init__(self, wavelet='haar', mode='symmetric'):
        super(FirstOrderWaveletTransform, self).__init__()
        self.wavelet = wavelet
        self.mode = mode

    def forward(self, x):
        # Check if input is 2D or 3D
        assert len(x.shape) == 4, "Input should be in 4D format"
        batch_size, channels, height, width = x.shape
        
        # Initialize lists to store the output components
        L_list = []
        H_list = []
        
        for b in range(batch_size):
            L_batch = []
            H_batch = []
            for c in range(channels):
                L, H = self._wavelet_transform_2d(x[b, c])  # Pass the correct slice (b, c)
                L_batch.append(L)
                H_batch.append(H)
                
            # Stack L and H for each batch
            L_list.append(torch.stack(L_batch, dim=0))  # Stack along channel dimension
            H_list.append(torch.stack(H_batch, dim=0))  # Stack along channel dimension
        
        # Stack L and H across batches
        L_output = torch.stack(L_list, dim=0)  # Stack along batch dimension
        H_output = torch.stack(H_list, dim=0)  # Stack along batch dimension

        return L_output, H_output

    def _wavelet_transform_2d(self, img):
        # Convert the input tensor to NumPy array
        img_np = img.cpu().numpy()

        # Perform the 2D Discrete Wavelet Transform (DWT)
        coeffs2 = pywt.dwt2(img_np, self.wavelet, mode=self.mode)

        # coeffs2 contains (cA, (cH, cV, cD)): 
        # cA = Approximation coefficients
        # cH = Horizontal detail coefficients
        # cV = Vertical detail coefficients
        # cD = Diagonal detail coefficients
        L, (cH, cV, cD) = coeffs2
        H = cH + cV + cD

        # Convert back to PyTorch tensors
        L = torch.from_numpy(L).to(img.device)
        H = torch.from_numpy(H).to(img.device)

        return L, H


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
        )
    
    def forward(self, x):
        return self.layer(x)


class Down(nn.Module):
    """Max pooling then double convolutions"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Down, self).__init__()
        self.down_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvLayer(in_channels, out_channels),
            ConvLayer(out_channels, out_channels)
        )

    def forward(self, x):
        return self.down_conv(x)


class Up(nn.Module):
    """Concatenate, then upsampling, then double convolution -->in_channels//2"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                ConvLayer(3*in_channels, 3*in_channels//2, 1, 1, 0)
            )
        else:
            self.up = nn.ConvTranspose2d(3*in_channels, 3*in_channels//2, kernel_size=2, stride=2)
        
        self.doubleconv = nn.Sequential(
            ConvLayer(3*in_channels//2, out_channels),
            ConvLayer(out_channels, out_channels)
        )

    def forward(self, x0, x1, x2):
        x = torch.cat([x0, x1, x2], dim=1)
        x = self.up(x)
        return self.doubleconv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.out_conv = ConvLayer(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return normalize(self.out_conv(x))


class Generator(nn.Module):
    def __init__(self, latent_dim, n_channels, n_classes, bilinear=True):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.wf = FirstOrderWaveletTransform()
        #for Low frequency information
        in_ch = self.n_channels
        self.conv1_1 = ConvLayer(in_ch, 32)
        self.conv2_1 = ConvLayer(32, 32)
        self.down1_1 = Down(32, 64)
        self.down2_1 = Down(64, 128)
        self.down3_1 = Down(128, 256)

        #for High frequency information
        self.conv1_2 = ConvLayer(in_ch, 32)
        self.conv2_2 = ConvLayer(32, 32)
        self.down1_2 = Down(32, 64)
        self.down2_2 = Down(64, 128)
        self.down3_2 = Down(128, 256)

        #for upsampling
        self.conv3 = ConvLayer(512, 256)
        self.up1 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.up3 = Up(64, 32)
        self.up4 = Up(32, 32)

        #ending
        self.conv_end0 = ConvLayer(32, 8)
        self.conv_end1 = ConvLayer(8, 8)
        self.conv_end2 = ConvLayer(8, 8)
        
        self.outc = OutConv(8, n_classes)

    def forward(self, x):
        x1, x2 = self.wf(x) #x1 is L, x2 is H
        #for Low frequency information
        x1 = self.conv1_1(x1)
        down0_1 = self.conv2_1(x1)
        down1_1 = self.down1_1(down0_1)
        down2_1 = self.down2_1(down1_1)
        down3_1 = self.down3_1(down2_1)

        #for High frequency information
        x2 = self.conv1_2(x2)
        down0_2 = self.conv2_2(x2)
        down1_2 = self.down1_2(down0_2)
        down2_2 = self.down2_2(down1_2)
        down3_2 = self.down3_2(down2_2)

        #concatenate and upsampling
        x0 = torch.cat((down3_1, down3_2), dim=1)
        x0 = self.conv3(x0)
        x0 = self.up1(x0, down3_1, down3_2)
        x0 = self.up2(x0, down2_1, down2_2)
        x0 = self.up3(x0, down1_1, down1_2)
        x0 = self.up4(x0, down0_1, down0_2)

        #ending
        x0 = self.conv_end0(x0)
        conv_end1 = self.conv_end1(x0)
        conv_end2 = self.conv_end2(x0 + conv_end1)

        x = self.outc(x0 + conv_end1 + conv_end2)
        
        return x.to(device)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.BatchNorm2d(64, momentum=0.85),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128, momentum=0.85),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(256, momentum=0.85),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(256 * 32 * 32, 1)
        )

    def forward(self, img):
        validity = self.model(img)
        #validity = normalize(validity)
        return validity

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        
        # Load a pre-trained VGG network
        self.vgg = models.vgg16(pretrained=True).features.to(device)
        
        # Freeze the VGG model weights
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        self.criterion = nn.L1Loss()
    
    def forward(self, input_img, target_img):
        assert input_img.shape == target_img.shape, "Input and target images must have the same shape"

        # Preprocess the images to match VGG's input requirements (Normalize, etc.)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_img = input_img.to(torch.device('cpu'))
        target_img = target_img.to(torch.device('cpu'))
        input_img = transform(input_img).to(device)
        target_img = transform(target_img).to(device)
        
        # Extract feature representations at specified layers
        input_features = self.vgg(input_img)
        target_features = self.vgg(target_img)
        
        # Compute the perceptual loss between input and target features
        loss = self.criterion(input_features, target_features)
        
        return loss

class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()
    
    def forward(self, x):
        # Horizontal and vertical differences
        diff_h = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
        diff_v = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
        # Total variation loss is the sum of these differences
        loss = diff_h.mean() + diff_v.mean()
        return loss
