import numpy as np
import cv2
import os
import pickle
import sys
import math

from PIL import Image
import matplotlib.pyplot as plt
from torchvision.io import read_image

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.optim as optim

#from tensorboardX import SummaryWriter

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton

torch.set_default_dtype(torch.float32)


class SkeToImageTransform:
    def __init__(self, image_size):
        self.imsize = image_size

    def __call__(self, ske):
        image = np.ones((self.imsize, self.imsize, 3), dtype=np.uint8) * 255
        ske.draw(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


class VideoSkeletonDataset(Dataset):
    def __init__(self, videoSke, ske_reduced, source_transform=None, target_transform=None):
        """ videoSkeleton dataset: 
                videoske(VideoSkeleton): video skeleton that associate a video and a skeleton for each frame
                ske_reduced(bool): use reduced skeleton (13 joints x 2 dim=26) or not (33 joints x 3 dim = 99)
        """
        self.videoSke = videoSke
        self.source_transform = source_transform
        self.target_transform = target_transform
        self.ske_reduced = ske_reduced
        print("VideoSkeletonDataset: ",
              "ske_reduced=", ske_reduced, "=(", Skeleton.reduced_dim, " or ", Skeleton.full_dim, ")" )

    def __len__(self):
        return self.videoSke.skeCount()

    def __getitem__(self, idx):
        reduced = True
        ske = self.videoSke.ske[idx]
        ske = self.preprocessSkeleton(ske)
        
        # Process image
        image = Image.open(self.videoSke.imagePath(idx))
        if self.target_transform:
            image = self.target_transform(image)
        return ske, image

    def preprocessSkeleton(self, ske):
        if self.source_transform:
            ske = self.source_transform(ske)
        else:
            ske = torch.from_numpy(ske.__array__(reduced=self.ske_reduced).flatten())
            ske = ske.to(torch.float32)
            ske = ske.reshape(ske.shape[0], 1, 1)
        return ske

    def tensor2image(self, normalized_image):
        numpy_image = normalized_image.detach().cpu().numpy()
        numpy_image = np.transpose(numpy_image, (1, 2, 0))
        numpy_image = cv2.cvtColor(np.array(numpy_image), cv2.COLOR_RGB2BGR)
        denormalized_image = numpy_image * 0.5 + 0.5
        return (denormalized_image * 255).astype(np.uint8)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class GenNNSkeToImage(nn.Module):
    """ Class that generates an image from a skeleton """
    def __init__(self):
        super(GenNNSkeToImage, self).__init__()
        self.input_dim = Skeleton.reduced_dim
        self.model = nn.Sequential(
            nn.ConvTranspose2d(self.input_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self.model.apply(init_weights)
        print(self.model)

    def forward(self, z):
        img = self.model(z)
        return img


class GenVanillaNN():
    """ Class that generates a new image from a skeleton posture """
    def __init__(self, videoSke, loadFromFile=False, optSkeOrImage=1):
        image_size = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.netG = GenNNSkeToImage().to(self.device)
        src_transform = None
        self.filename = 'data/Dance/DanceGenVanillaFromSke.pth'

        tgt_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, target_transform=tgt_transform, source_transform=src_transform)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=16, shuffle=True)
        
        if loadFromFile and os.path.isfile(self.filename):
            print("GenVanillaNN: Load=", self.filename)
            state_dict = torch.load(self.filename)  # Chargement des poids seulement
            self.netG.load_state_dict(state_dict)
        else:
            print("GenVanillaNN: No pre-trained model loaded, starting training...")
            self.train(100)

    def train(self, n_epochs=20):
        """ Entraîne le réseau de neurones pour générer des images à partir de squelettes """
        self.netG.train()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        for epoch in range(n_epochs):
            for i, data in enumerate(self.dataloader, 0):
                ske, target = data
                ske, target = ske.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                output = self.netG(ske)
                loss = criterion(output, target)
                
                loss.backward()
                optimizer.step()
                
                if i % 100 == 0:
                    print(f'Epoch [{epoch+1}/{n_epochs}], Step [{i+1}/{len(self.dataloader)}], Loss: {loss.item():.4f}')
                    
            # Sauvegarde du modèle après chaque époque
        print("Training completed. Saving the model to:", self.filename)
        torch.save(self.netG.state_dict(), self.filename)
        print("Model saved.")


    def generate(self, ske):
        """ Génère une image à partir d'un squelette """
        self.netG.eval()
        with torch.no_grad():
            ske_t = self.dataset.preprocessSkeleton(ske).to(self.device)
            ske_t_batch = ske_t.unsqueeze(0)
            normalized_output = self.netG(ske_t_batch)
            res = self.dataset.tensor2image(normalized_output[0])
        return res


if __name__ == '__main__':
    force = False
    optSkeOrImage = 2
    n_epoch = 2000
    training = True
    
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            force = sys.argv[2].lower() == "true"
    else:
        filename = "dance/data/taichi1.mp4"
        
    targetVideoSke = VideoSkeleton(filename)

    if training:
        gen = GenVanillaNN(targetVideoSke, loadFromFile=False)
        gen.train(n_epoch)
    else:
        gen = GenVanillaNN(targetVideoSke, loadFromFile=True)

    for i in range(targetVideoSke.skeCount()):
        image = gen.generate(targetVideoSke.ske[i])
        nouvelle_taille = (256, 256)
        image = cv2.resize(image, nouvelle_taille)
        cv2.imshow('Image', image)
        key = cv2.waitKey(-1)
