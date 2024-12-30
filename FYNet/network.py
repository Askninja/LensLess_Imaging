# Generator and Discriminator networks
# as well as loss funtions used
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
import torchvision
import scipy
import matplotlib.pyplot as plt

from PIL import Image
import skimage

def normalize(x):
    max_value = x.max()
    min_value = x.min()
    return (x - min_value) / (max_value - min_value)

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir_1, input_dir_2, target_dir, transform=None):
        self.input_dir_1 = input_dir_1
        self.input_dir_2 = input_dir_2
        self.target_dir = target_dir
        self.input_images_1 = sorted(os.listdir(input_dir_1))  # Sort to align input and target
        self.input_images_2 = sorted(os.listdir(input_dir_2))
        self.target_images = sorted(os.listdir(target_dir))
        self.transform = transform

    def __len__(self):
        return len(self.target_images)

    def __getitem__(self, idx):
        # Load input and target images
        input_image_path_1 = os.path.join(self.input_dir_1, self.input_images_1[idx])
        input_image_path_2 = os.path.join(self.input_dir_2, self.input_images_2[idx])
        target_image_path = os.path.join(self.target_dir, self.target_images[idx])
        
        input_image_1 = Image.open(input_image_path_1)
        input_image_2 = Image.open(input_image_path_2)
        target_image = Image.open(target_image_path)
        
        # Apply transformations
        if self.transform:
            input_image_1 = self.transform(input_image_1)
            input_image_2 = self.transform(input_image_2)
            target_image = self.transform(target_image)
            
        return input_image_1, input_image_2, target_image

class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(1, out_channels, self.modes, self.modes, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(1, out_channels, self.modes, self.modes, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes, :self.modes] = \
            self.compl_mul2d(x_ft[:, :, :self.modes, :self.modes], self.weights1)
        out_ft[:, :, -self.modes:, :self.modes] = \
            self.compl_mul2d(x_ft[:, :, -self.modes:, :self.modes], self.weights2)

        # out_ft = self.ca(out_ft.abs()) * out_ft
        # out_ft = self.sa(out_ft.abs()) * out_ft

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


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

class FourierBlock(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(FourierBlock, self).__init__()

        """
        SpectralConv2d_fast -> PReLU -> SpectralConv2d (of same modes and channels) and 2 conv layers
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        self.sconv1 = SpectralConv2d_fast(self.in_channels, self.in_channels, modes)
        self.prelu = nn.PReLU()
        self.sconv2 = SpectralConv2d_fast(self.out_channels, self.in_channels, modes)

        self.conv1 = ConvLayer(self.in_channels, self.out_channels)
        self.conv2 = ConvLayer(self.in_channels, self.out_channels)

    def forward(self, x):
        ft1 = self.sconv1(x)
        prelu = self.prelu(ft1)
        ft2 = self.sconv2(x + prelu)

        conv1 = self.conv1(x)
        conv2 = self.conv2(ft2)

        return (conv1 + conv2)


class Up(nn.Module):
    """Concatenate, then upsampling, then double convolution -->in_channels//2"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                ConvLayer(in_channels, in_channels//2, 1, 1, 0)
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.up(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.out_conv = ConvLayer(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return normalize(self.out_conv(x))
        #return self.out_conv(x)


class Generator(nn.Module):
    def __init__(self, latent_dim, n_channels, n_classes, bilinear=True):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        #self.threshold = nn.Parameter(torch.tensor(0.1, requires_grad=True))

        #for Input 1
        in_ch = self.n_channels
        self.conv1_1 = ConvLayer(in_ch, 32)
        self.conv2_1 = ConvLayer(32, 32)

        #for Input 2
        self.conv1_2 = ConvLayer(in_ch, 32)
        self.conv2_2 = ConvLayer(32, 32)
        
        #downsampling
        self.down = nn.MaxPool2d(2)
        self.fb1 = FourierBlock(64, 64, self.latent_dim//4)
        self.fb2 = FourierBlock(64, 64, self.latent_dim//4)

        #upsampling
        self.up = Up(64, 32, bilinear)
        self.conv3 = ConvLayer(32, 32)
        self.conv4 = ConvLayer(32, 32)

        self.fb3 = FourierBlock(64, 64, self.latent_dim//4)
        self.fb4 = FourierBlock(64, 32, self.latent_dim//4)

        #ending
        self.conv_end0 = ConvLayer(32*3, 8)
        self.conv_end1 = ConvLayer(8, 8)
        self.conv_end2 = ConvLayer(8, 8)
        
        self.outc = OutConv(8, n_classes)

    def forward(self, x1, x2):
        #for Input 1
        x1 = self.conv1_1(x1)
        x1 = self.conv2_1(x1)
        down0_1 = self.down(x1)

        #for Input 2
        x2 = self.conv1_2(x2)
        x2 = self.conv2_2(x2)
        down0_2 = self.down(x2)

        #concatenate
        fb1 = torch.cat((down0_1, down0_2), dim=1)
        fb1 = self.fb1(fb1)
        fb1 = self.fb2(fb1)

        #upsampling
        up = self.up(fb1)
        conv3 = self.conv3(up)
        conv4 = self.conv4(conv3)
        x0 = torch.cat((up, conv4), dim=1)
        x0 = self.fb3(x0)
        x0 = self.fb4(x0)

        #ending
        x0 = torch.cat((x0, x1, x2), dim=1)
        x0 = self.conv_end0(x0)
        conv_end1 = self.conv_end1(x0)
        conv_end2 = self.conv_end2(x0 + conv_end1)

        out = self.outc(x0 + conv_end1 + conv_end2)
        
        return out.to(device)


# Define the discriminator receiving input of [N, 2, W, H], N is batch size
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
            nn.Linear(256 * 16 * 16, 1)
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
    # Total variation loss
    def __init__(self):
        super(TVLoss, self).__init__()
    
    def forward(self, x):
        # Horizontal and vertical differences
        diff_h = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
        diff_v = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
        # Total variation loss is the sum of these differences
        loss = diff_h.mean() + diff_v.mean()
        return loss

class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1  #grayscale
    
    def create_window(self, channel: int):
        _1D_window = torch.hann_window(self.window_size, periodic=False).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, self.window_size, self.window_size).contiguous()
        return window

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        window = self.create_window(channel).to(device)
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return ssim_map.mean()

        return ssim_map.mean(1).mean(1).mean(1)
