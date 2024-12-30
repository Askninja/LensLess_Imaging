# Use GAN with generator is FYNet
# Input data are 40x and 60x

import os
import glob
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

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
from network import ImageDataset, Generator, Discriminator, PerceptualLoss, TVLoss, SSIM

from PIL import Image
import skimage

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(0)
np.random.seed(0)

latent_dim = 256
batch_size = 18

lr = 1e-4
num_epochs = 10

train_loss_hist = []
val_loss_hist = []
d_loss_hist = []

def normalize(x):
    max_value = x.max()
    min_value = x.min()
    return (x - min_value) / (max_value - min_value)

def main():
    #===================== LOADING DATA ===========================
    path_model = 'models/iscat_new/FYGAN_no-norm'
    plot_output = 'outputs/plots_no-norm'

    if not os.path.exists(path_model):
        os.makedirs(path_model)
    if not os.path.exists(plot_output):
        os.makedirs(plot_output)
    
    # Create dataset
    location = 'noise_dataset/'
    train_input_dir_1 = location+'train/Mag_40'
    train_input_dir_2 = location+'train/Mag_60'
    train_target_dir = location+'train/GT'
    
    val_input_dir_1 = location+'val/Mag_40'
    val_input_dir_2 = location+'val/Mag_60'
    val_target_dir = location+'val/GT'

    train_dataset = ImageDataset(input_dir_1=train_input_dir_1, input_dir_2=train_input_dir_2, target_dir=train_target_dir, transform=transform)
    val_dataset = ImageDataset(input_dir_1=val_input_dir_1, input_dir_2=val_input_dir_2, target_dir=val_target_dir, transform=transform)

    # Create DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    # Initialize generator and discriminator
    generator = Generator(latent_dim, 1, 1).to(device)
    discriminator = Discriminator().to(device)
    # Loss functions
    adversarial_loss = nn.BCEWithLogitsLoss().to(device)
    mse_loss = nn.MSELoss().to(device)
    bce_loss = nn.BCEWithLogitsLoss().to(device)
    perceptual_loss = PerceptualLoss().to(device)
    ssim_loss = SSIM().to(device)
    tv_loss = TVLoss().to(device)
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, weight_decay=1e-5) #regularization
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, weight_decay=1e-5)
    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=5, gamma=0.1)
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=5, gamma=0.1)

    start_ep = 0
    min_val_loss = 1e10
    """
    #loading Generator
    if os.path.isfile(os.path.join(path_model,"checkpoint_G.pth")):
        checkpoint = torch.load(os.path.join(path_model,"checkpoint_G.pth"),map_location=torch.device('cpu'))
        start_ep = checkpoint['epoch']
        print("Resuming from the checkpoint: ep", start_ep+1)
        np.random.set_state(checkpoint['np_rand_state'])
        torch.set_rng_state(checkpoint['torch_rand_state'])
        optimizer_G.load_state_dict(checkpoint['optimizer'])
        scheduler_G.load_state_dict(checkpoint['scheduler'])
        generator.load_state_dict(checkpoint['model'])

    #loading Discriminator
    if os.path.isfile(os.path.join(path_model,"checkpoint_D.pth")):
        checkpoint = torch.load(os.path.join(path_model,"checkpoint_D.pth"),map_location=torch.device('cpu'))
        start_ep = checkpoint['epoch']
        print("Resuming from the checkpoint: ep", start_ep+1)
        np.random.set_state(checkpoint['np_rand_state'])
        torch.set_rng_state(checkpoint['torch_rand_state'])
        optimizer_D.load_state_dict(checkpoint['optimizer'])
        scheduler_D.load_state_dict(checkpoint['scheduler'])
        discriminator.load_state_dict(checkpoint['model'])
    """
    # Training loop
    for epoch in range(num_epochs):
        for i, (x1,x2,yy) in enumerate(train_loader):
            #data and its label
            x1 = x1.to(device)
            x2 = x2.to(device)
            yy = yy.to(device)
            
            # Adversarial ground truths
            valid = torch.ones(x1.shape[0], 1, device=device)
            fake = torch.zeros(x1.shape[0], 1, device=device)
    
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            # Generate a batch of images
            fake_images = generator(x1, x2)
    
            # Measure discriminator's ability to classify real and fake images
            real_loss = adversarial_loss(discriminator(yy), valid)
            fake_loss = adversarial_loss(discriminator(fake_images.detach()), fake)
            d_loss = (real_loss+fake_loss)/2
            # Backward pass and optimize
            d_loss.backward()
            optimizer_D.step()
    
            # -----------------
            #  Train Generator
            # -----------------
    
            optimizer_G.zero_grad()
            # Generate a batch of images
            gen_images = generator(x1, x2)
            # Total loss
            g_loss = 0.2*adversarial_loss(discriminator(gen_images), valid) + bce_loss(gen_images, yy) + perceptual_loss(gen_images, yy) + 0.05*tv_loss(gen_images)
            
            # Backward pass and optimize
            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Progress Monitoring
            # ---------------------
            if (i + 1) % 50 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}]\
                            Batch {i+1}/{len(train_loader)} "
                    f"Discriminator Loss: {d_loss.item():.4f} "
                    f"Generator Loss: {g_loss.item():.4f}"
                )
                
        #======================VALIDATION=======================
        valid_ssim = 0
        with torch.no_grad():
            for i, (x1, x2, yy) in enumerate(val_loader):
                x1, x2, yy = x1.to(device), x2.to(device), yy.to(device)
                im = generator(x1, x2)
                valid_ssim += 1-(ssim_loss(im, yy)).item()
        valid_ssim /= len(val_loader)
        print(
            f"Epoch [{epoch+1}/{num_epochs}]\
                    Valid SSIM Loss: {valid_ssim:.4f} "
        )
        
        if valid_ssim < min_val_loss:
            min_val_loss = valid_ssim
            #save generator
            torch.save({'epoch': epoch+start_ep,
                        'model': generator.state_dict(),
                        'loss': g_loss.item(),
                        'optimizer': optimizer_G.state_dict(),
                        'scheduler': scheduler_G.state_dict(),
                        'np_rand_state': np.random.get_state(),
                        'torch_rand_state': torch.get_rng_state(),
                        }, os.path.join(path_model,"checkpoint_FYGAN_final.pth"))
            #save discriminator
            torch.save({'epoch': epoch+start_ep,
                        'model': discriminator.state_dict(),
                        'loss': d_loss.item(),
                        'optimizer': optimizer_D.state_dict(),
                        'np_rand_state': np.random.get_state(),
                        'torch_rand_state': torch.get_rng_state(),
                        }, os.path.join(path_model,"checkpoint_D_final.pth"))
        #save losses
        d_loss_hist.append(d_loss.item())
        train_loss_hist.append(g_loss.item())
        val_loss_hist.append(valid_ssim)
        #save generator
        torch.save({'epoch': epoch+start_ep,
                    'model': generator.state_dict(),
                    'loss': g_loss.item(),
                    'optimizer': optimizer_G.state_dict(),
                    'scheduler': scheduler_G.state_dict(),
                    'np_rand_state': np.random.get_state(),
                    'torch_rand_state': torch.get_rng_state(),
                    }, os.path.join(path_model,f"checkpoint_FYGAN_ep={epoch}.pth"))
        #save discriminator
        torch.save({'epoch': epoch+start_ep,
                    'model': discriminator.state_dict(),
                    'loss': d_loss.item(),
                    'optimizer': optimizer_D.state_dict(),
                    'scheduler': scheduler_D.state_dict(),
                    'np_rand_state': np.random.get_state(),
                    'torch_rand_state': torch.get_rng_state(),
                    }, os.path.join(path_model,f"checkpoint_D_ep={epoch}.pth"))

    
    #visualize losses
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # Plot Discrimination Loss
    axs[0].plot(range(1, num_epochs + 1), d_loss_hist, marker='o', linestyle='-', color='r')
    axs[0].set_title('Discrimination Loss vs. Epochs')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Discrimination Loss')
    axs[0].grid(True)

    # Plot Generation Loss
    axs[1].plot(range(1, num_epochs + 1), train_loss_hist, marker='o', linestyle='-', color='b')
    axs[1].set_title('Generation Loss vs. Epochs')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Generation Loss')
    axs[1].grid(True)

    # Plot Validation and Test Loss
    axs[2].plot(range(1, num_epochs + 1), val_loss_hist, marker='o', linestyle='-', color='g')
    axs[2].set_title('Validation Loss vs. Epochs')
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('Generation Loss')
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig(plot_output+f'/FYGAN_ep={num_epochs}.png')
    plt.show()

if __name__ == '__main__':
    main()
