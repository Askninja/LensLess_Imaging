"Choose an image from dataset and generate output"
import glob
import os
import random
import numpy as np
import scipy
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from network import normalize, Generator
from wnet import device

import torch
import torch.nn as nn
from torchvision import transforms

from PIL import Image
from skimage.metrics import structural_similarity as ssim


latent_dim = 512
batch_size = 1

def add_gaussian_noise(image, mean=0, std=10):
    image_np = np.array(image)
    noise = np.random.normal(mean, std, image_np.shape).astype(np.uint8)
    noisy_image_np = np.clip(image_np + 0.05*noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image_np)

def loadData(inp_fname, gt_fname=None, transform=None, noise=None):
    input_image = Image.open(inp_fname)
    #Add noise
    if noise:
        input_image = add_gaussian_noise(input_image, 0, 10)
    # Apply transformations
    if transform:
        input_image = transform(input_image)
    
    if gt_fname:
        target_image = Image.open(gt_fname)
        # Apply transformations
        if transform:
            target_image = transform(target_image)
        return input_image, target_image
        
    return input_image


# Define transformations
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x[0].unsqueeze(0).unsqueeze(0))
    #transforms.Lambda(lambda x: 1-x)
])

#path_model = 'Desktop/URECA/iscat_results/models/FNet'
path_model = 'Desktop/URECA/GAN-main/models/WNet'
path_output = 'Desktop/URECA/iscat_results/outputs/WNet'
if not os.path.exists(path_output):
    os.makedirs(path_output)

location = 'Input/'
inp_image = 'Desktop/URECA/iSCAT_data/' + location #preprocessed with Gaussian noise except test
gt_image = 'Desktop/URECA/iSCAT_data/GT/GT_0002.png'
fname = 'Input_0002.png'

model = FNet(latent_dim, 1, 1).to(device) #generate
#loading model
checkpoint_G = torch.load(os.path.join(path_model, "checkpoint_WNet_final.pth"), map_location='cpu')
start_ep = checkpoint_G['epoch']
print("Resuming from the checkpoint: ep", start_ep+1)
model.load_state_dict(checkpoint_G['model'], strict=False)

#reconstruct and count
if gt_image:
    xx, yy = loadData(inp_image+fname, gt_image, transform, True)
    xx = xx.to(device)
    yy = yy.to(device)
    
    #reconstruc image
    im = model(xx).detach()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    #display input image
    inp = xx.squeeze()
    inp = inp.cpu().numpy()
    axes[0].imshow(inp, cmap='gray')
    axes[0].set_title("Input")
    
    #display ground truth image
    gt = yy.squeeze()
    gt = gt.cpu().numpy()
    axes[1].imshow(gt, cmap='gray')
    axes[1].set_title("GT")
    
    #display output images
    out = im.squeeze()
    out = out.cpu().numpy()
    axes[2].imshow(out, cmap='gray')
    re_loss = ssim(out, gt, data_range=1.0) #SSIM index of reconstruction
    axes[2].set_title(f"SSIM: {re_loss:.4f}")
    
    plt.tight_layout()
    plt.savefig(path_output + f"/result_{fname}.png")
    plt.show()
    
else:
    xx = loadData(inp_image+fname, transform=transform)
    xx = xx.to(device)
    #reconstruc image
    im = model(xx).detach()
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    #display input image
    inp = xx.squeeze()
    inp = inp.cpu().numpy()
    axes[0].imshow(inp, cmap='gray')
    axes[0].set_title("Input")
    
    #display output images
    out = im.squeeze()
    out = out.cpu().numpy()
    axes[1].imshow(out, cmap='gray')
    axes[1].set_title("Reconstruction")
    
    plt.tight_layout()
    plt.savefig(path_output + f"/Result_Test_{fname}")
    plt.show()
    
