
import numpy as np
import cv2
import os
import pickle
import sys
import math

import matplotlib.pyplot as plt

from torchvision.io import read_image
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton
from GenVanillaNN import * 



class Discriminator(nn.Module):
    def __init__(self, ngpu=0):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input)
    



class GenGAN():
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
    """
    def __init__(self, videoSke, loadFromFile=False):
        self.netG = GenNNSkeToImage()
        self.netD = Discriminator()
        self.real_label = 1.
        self.fake_label = 0.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.filename = 'data/Dance/DanceGenGAN.pth'
        tgt_transform = transforms.Compose(
                            [transforms.Resize((64, 64)),
                            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            transforms.CenterCrop(64),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])
        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, target_transform=tgt_transform)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=32, shuffle=True)
        if loadFromFile and os.path.isfile(self.filename):
            print("GenGAN: Load=", self.filename, "   Current Working Directory=", os.getcwd())
            self.netG = torch.load(self.filename, map_location=self.device)


    def train(self, n_epochs=20):
        # Set up optimizers
        optimizerD = optim.Adam(self.netD.parameters(), lr=0.001, betas=(0.5, 0.999))
        optimizerG = optim.Adam(self.netG.parameters(), lr=0.001, betas=(0.5, 0.999))
        
        criterion = nn.BCELoss()  # Loss function
        MSE_loss = nn.MSELoss()  # Mean Squared Error loss
        
        # Move models to GPU if available
        self.netG.to(self.device)
        self.netD.to(self.device)

        # Training loop
        for epoch in range(n_epochs):
            for i, data in enumerate(self.dataloader):
                
                ############################
                # (1) Update Discriminator
                ###########################
                self.netD.zero_grad()
                
                # Train with real images
                real_images = data[1].to(self.device)  # Get real images from dataloader
                batch_size = real_images.size(0)
                label = torch.full((batch_size,), self.real_label, device=self.device)
                
                output = self.netD(real_images).view(-1)
                errD_real = criterion(output, label)
                errD_real.backward()
                
                # Train with fake images
                # fake = torch.randn(batch_size, Skeleton.reduced_dim, 1, 1, device=device)
                fake_images = self.netG(data[0].to(self.device))
                label.fill_(self.fake_label)
                
                output = self.netD(fake_images.detach()).view(-1)
                errD_fake = criterion(output, label)
                errD_fake.backward()
                optimizerD.step()
                
                ############################
                # (2) Update Generator
                ###########################
                self.netG.zero_grad()
                label.fill_(self.real_label)  # Fake labels are real for generator cost
                
                output = self.netD(fake_images).view(-1)
                errG_adv  = criterion(output, label)
                errG_mse = MSE_loss(fake_images, real_images)
                
                errG = errG_adv + 300*errG_mse
                errG.backward()
                optimizerG.step()
                
                # Logging and printing
                print(f'[{epoch}/{n_epochs}][{i}/{len(self.dataloader)}] '
                      f'Loss_D: {errD_real.item() + errD_fake.item():.4f} '
                      f'Loss_G: {errG.item():.4f}')
            
            # Save the generator model after each epoch
            torch.save(self.netG, self.filename)
            print(f'Saved model after epoch {epoch+1}/{n_epochs} to {self.filename}')




    def generate(self, ske):
        """ generator of image from skeleton """
        # Get the same device that the model is on
        device = next(self.netG.parameters()).device
        
        # Convert skeleton to tensor and move to correct device
        ske_t = torch.from_numpy(ske.__array__(reduced=True).flatten())
        ske_t = ske_t.to(torch.float32)
        ske_t = ske_t.reshape(1, Skeleton.reduced_dim, 1, 1)
        ske_t = ske_t.to(device)  # Move input tensor to same device as model
        
        # Generate image
        with torch.no_grad():  # Add this for inference
            normalized_output = self.netG(ske_t)
            # Move output back to CPU for further processing
            normalized_output = normalized_output.cpu()
        
        res = self.dataset.tensor2image(normalized_output[0])
        return res




if __name__ == '__main__':
    force = False
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            force = sys.argv[2].lower() == "true"
    else:
        filename = "data/taichi1.mp4"
    print("GenGAN: Current Working Directory=", os.getcwd())
    print("GenGAN: Filename=", filename)

    targetVideoSke = VideoSkeleton(filename)

    #if False:
    if True:    # train or load
        # Train
        gen = GenGAN(targetVideoSke, False)
        gen.train(100) #5) #200)
    else:
        gen = GenGAN(targetVideoSke, loadFromFile=True)    # load from file        


    # for i in range(targetVideoSke.skeCount()):
    #     image = gen.generate(targetVideoSke.ske[i])
    #     #image = image*255
    #     nouvelle_taille = (256, 256) 
    #     image = cv2.resize(image, nouvelle_taille)
    #     cv2.imshow('Image', image)
    #     key = cv2.waitKey(-1)

