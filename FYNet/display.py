# Display output of the model
import glob
import os
import random
import numpy as np
import scipy
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from numpy.lib import imag

from network import normalize, Generator
from fygan import device

import torch
import torch.nn as nn
from torchvision import transforms

from PIL import Image
from skimage.metrics import structural_similarity as ssim


latent_dim = 256
batch_size = 1

def loadData(input_dir_1, input_dir_2, target_dir, idx, transform=None):
    input_image_path_1 = os.path.join(input_dir_1, f"Input_{(idx):04d}.png")
    input_image_path_2 = os.path.join(input_dir_2, f"Input_{(idx):04d}.png")
    target_image_path = os.path.join(target_dir, f"GT_{(idx):04d}.png")
    
    input_image_1 = Image.open(input_image_path_1)
    input_image_2 = Image.open(input_image_path_2)
    target_image = Image.open(target_image_path)
    
    # Apply transformations
    if transform:
        input_image_1 = transform(input_image_1)
        input_image_2 = transform(input_image_2)
        target_image = transform(target_image)
        
    return input_image_1, input_image_2, target_image

# Define transformations
transform = transforms.Compose([
    transforms.Resize((latent_dim, latent_dim)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.unsqueeze(0)),
])

path_model = 'models/iscat_new/FYGAN-noise/checkpoint_FYGAN_final.pth'
path_output = 'outputs/new_noise'
dataset = 'train/'
input_dir_1 = 'noise_dataset/' + dataset + 'Mag_40'
input_dir_2 = 'noise_dataset/' + dataset + 'Mag_60'
target_dir = 'noise_dataset/' + dataset + 'GT'

index = 1748 #index of image
train = [1, 857, 1351, 2409, 2513, 4971]
val = [921, 1410, 1815, 3050]
test = [60, 3444]

model = Generator(latent_dim, 1, 1).to(device) #for YGAN

checkpoint_G = torch.load(path_model, map_location='cpu')
start_ep = checkpoint_G['epoch']
print("Resuming from the checkpoint: ep", start_ep)
model.load_state_dict(checkpoint_G['model'], strict=False)

#reconstruct and count
x1, x2, yy = loadData(input_dir_1, input_dir_2, target_dir, index, transform)
x1 = x1.to(device)
x2 = x2.to(device)

yy = yy.to(device)

im = normalize(model(x1, x2).detach())

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

#display input1 images
inp1 = x1.squeeze()
inp1 = inp1.cpu().numpy()
axes[0][0].imshow(inp1, cmap='gray')
axes[0][0].set_title("Mag_40")

#display input2 images
inp2 = x2.squeeze()
inp2 = inp2.cpu().numpy()
axes[0][1].imshow(inp2, cmap='gray')
axes[0][1].set_title("Mag_60")

#display ground truth images
gt = yy.squeeze()
gt = gt.cpu().numpy()
axes[1][0].imshow(gt, cmap='gray')
axes[1][0].set_title(f"GT")

#display output images
out = im.squeeze()
out = out.cpu().numpy()
axes[1][1].imshow(out, cmap='gray')
re_loss = ssim(out, gt, data_range=1.0) #SSIM index of reconstruction
axes[1][1].set_title(f"Result SSIM: {re_loss:.4f}")

plt.tight_layout()
plt.savefig(path_output + f"/FYGAN_{index:04d}.png")
plt.show()
