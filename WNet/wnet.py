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
from network import ImageDataset, Generator, Discriminator, PerceptualLoss, TVLoss

from PIL import Image
import skimage
from torchvision import transforms
import torchvision

torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

latent_dim = 512
batch_size = 8

lr = 1e-4
num_epochs = 28
train_loss_hist = []
d_loss_hist = []
val_loss_hist = []
min_val_loss = 1e10

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor()
])

def main():
    #===================== LOADING DATA ===========================
    path_model = 'models/iscat/WGANet'
    plot_output = 'outputs/plots_try'
    if not os.path.exists(path_model):
        os.makedirs(path_model)
    if not os.path.exists(plot_output):
        os.makedirs(plot_output)
    
    # Create dataset
    train_input_dir = 'iSCAT_processed/train/Input'
    train_target_dir = 'iSCAT_processed/train/GT'
    
    val_input_dir = 'iSCAT_processed/val/Input'
    val_target_dir = 'iSCAT_processed/val/GT'
    
    train_dataset = ImageDataset(input_dir=train_input_dir, target_dir=train_target_dir, transform=transform)
    val_dataset = ImageDataset(input_dir=val_input_dir, target_dir=val_target_dir, transform=transform)
    
    # Create DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
   
    # Initialize generator and discriminator
    generator = Generator(latent_dim, 1, 1).to(device)
    discriminator = Discriminator().to(device)
    # Loss functions
    adversarial_loss = nn.BCEWithLogitsLoss().to(device)
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    perceptual_loss = PerceptualLoss()
    tv_loss = TVLoss().to(device)
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, weight_decay=1e-5)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, weight_decay=1e-5)
    #Schedulers
    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=5, gamma=0.1)
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=5, gamma=0.1)

    start_ep = 1
    ran_epoch = 0
    min_val_loss = 1e10
    """
    #loading Generator's Model
    if os.path.isfile(os.path.join(path_model,"checkpoint_WNet_final.pth")):
        checkpoint = torch.load(os.path.join(path_model,"checkpoint_WNet_final.pth"),map_location=torch.device('cpu'))
        start_ep += checkpoint['epoch']
        print("Resuming from the checkpoint: ep", start_ep)
        np.random.set_state(checkpoint['np_rand_state'])
        torch.set_rng_state(checkpoint['torch_rand_state'])
        optimizer_G.load_state_dict(checkpoint['optimizer'])
        scheduler_G.load_state_dict(checkpoint['scheduler'])
        generator.load_state_dict(checkpoint['model'])

    if os.path.isfile(os.path.join(path_model,"checkpoint_D_final.pth")):
        checkpoint = torch.load(os.path.join(path_model,"checkpoint_D_final.pth"),map_location=torch.device('cpu'))
        print("Resuming from the checkpoint: ep", start_ep)
        optimizer_D.load_state_dict(checkpoint['optimizer'])
        scheduler_D.load_state_dict(checkpoint['scheduler'])
        discriminator.load_state_dict(checkpoint['model'])
    """
    # Training loop
    for epoch in range(num_epochs):
        ran_epoch += 1
        for i, (xx, yy) in enumerate(train_loader):
            #data and its label
            xx = xx.to(device)
            yy = yy.to(device)
            # Adversarial ground truths
            valid = torch.ones(xx.shape[0], 1, device=device)
            fake = torch.zeros(xx.shape[0], 1, device=device)
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            # Generate a batch of images
            fake_images = generator(xx)
    
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
            # Total loss
            gen_images = generator(xx)
            g_loss = 0.1*adversarial_loss(discriminator(gen_images), valid) + bce_loss(gen_images, yy) + perceptual_loss(gen_images, yy) #+ 0.05*tv_loss(gen_images)
            
            # Backward pass and optimize
            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Progress Monitoring
            # ---------------------
            if (i + 1) % 125 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}]\
                            Batch {i+1}/{len(train_loader)} "
                    f"Discriminator Loss: {d_loss.item():.4f} "
                    f"Generator Loss: {g_loss.item():.4f}"
                )
        #======================VALIDATION=======================
        valid_mse = 0
        with torch.no_grad():
            for i, (xx, yy) in enumerate(val_loader):
                xx, yy = xx.to(device), yy.to(device)
                im = generator(xx)
                valid_mse += (mse_loss(im, yy)).item()
        valid_mse /= len(val_loader)
        
        print(
            f"Epoch [{epoch+1}/{num_epochs}]\
                    Valid MSE Loss: {valid_mse:.4f} "
        )

        #save model
        torch.save({'epoch': epoch+start_ep,
                    'model': generator.state_dict(),
                    'optimizer': optimizer_G.state_dict(),
                    'scheduler': scheduler_G.state_dict(),
                    'np_rand_state': np.random.get_state(),
                    'torch_rand_state': torch.get_rng_state(),
                    }, os.path.join(path_model,f"checkpoint_WNet_ep={epoch+start_ep}.pth"))
        torch.save({'epoch': epoch+start_ep,
                    'model': discriminator.state_dict(),
                    'optimizer': optimizer_D.state_dict(),
                    'scheduler': scheduler_D.state_dict(),
                    }, os.path.join(path_model,f"checkpoint_D_ep={epoch+start_ep}.pth"))
        
        if (epoch + 1) % 1 == 0:
            #save losses
            d_loss_hist.append(d_loss.item())
            train_loss_hist.append(g_loss.item())
            val_loss_hist.append(valid_mse)
            if valid_mse < min_val_loss:
                #save model based on validation loss
                min_val_loss = valid_mse
                #max_g_loss = g_loss.item()
                torch.save({'epoch': epoch+start_ep,
                            'model': generator.state_dict(),
                            'optimizer': optimizer_G.state_dict(),
                            'scheduler': scheduler_G.state_dict(),
                            'np_rand_state': np.random.get_state(),
                            'torch_rand_state': torch.get_rng_state(),
                            }, os.path.join(path_model,"checkpoint_WNet_final.pth"))
                torch.save({'epoch': epoch+start_ep,
                            'model': discriminator.state_dict(),
                            'optimizer': optimizer_D.state_dict(),
                            'scheduler': scheduler_D.state_dict(),
                            }, os.path.join(path_model,"checkpoint_D_final.pth"))
        
            
    #visualize losses
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # Plot Discriminator Loss
    axs[0].plot(range(start_ep, ran_epochs + start_ep), d_loss_hist, marker='o', linestyle='-', color='b')
    axs[0].set_title('Discriminator Loss vs. Epochs')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].grid(True)

    # Plot Generation Loss
    axs[1].plot(range(start_ep, ran_epochs + start_ep), train_loss_hist, marker='o', linestyle='-', color='g')
    axs[1].set_title('Train Loss vs. Epochs')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Loss')
    axs[1].grid(True)

    # Plot Val Loss
    axs[2].plot(range(start_ep, ran_epochs + start_ep), val_loss_hist, marker='o', linestyle='-', color='b')
    axs[2].set_title('Val Loss vs. Epochs')
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('Loss')
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig(plot_output+f'/WNet_eps={num_epochs}.png')
    plt.show()

if __name__ == '__main__':
    main()
