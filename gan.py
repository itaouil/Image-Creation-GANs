#!/usr/bin/env python

# Deep Convolutional GANs

# Compatible with Python2
from __future__ import print_function

# Import libraries
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

# Import classes
from modules.generator import G
from modules.discriminator import D

# Import config file
from config import config as cfg

# Hyperparameters settings
batchSize = cfg.data['batchSize'] # We set the size of the batch.
imageSize = cfg.data['imageSize'] # We set the size of the generated images (64x64).

# Creating the transformations
transform = transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]) # We create a list of transformations (scaling, tensor conversion, normalization) to apply to the input images.

# Loading the dataset
dataset = dset.CIFAR10(root = './data', download = True, transform = transform) # We download the training set in the ./data folder and we apply the previous transformations on each image.
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle = True, num_workers = 2) # We use dataLoader to get the images of the training set batch by batch.

# Defining the weights_init function that takes as input a neural network m and that will initialize all its weights.
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# Generator object
netG = G()
netG.apply(weights_init)

# Discriminator object
netD = D()
netD.apply(weights_init)

""" GAN's training """
# loss funtion
criterion = nn.BCELoss()
# Discriminator optimiser
optimiserD = optim.Adam(netD.parameters(), lr = 0.0002, betas = (0.5, 0.999))
# Generator optimiser
optimiserG = optim.Adam(netG.parameters(), lr = 0.0002, betas = (0.5, 0.999))

# Training process
for epoch in range(25):
    for i, data in enumerate(dataloader, 0):

        # Initialise grad with
        # respect to the weights
        # to 0
        netD.zero_grad()

        # Train discriminator (real image)
        real, _ = data
        torch_input = Variable(real)
        target = Variable(torch.ones(torch_input.size()[0]))
        output = netD(torch_input) # Prediction (0, 1)
        errD_real = criterion(output, target)

        # Train discriminator (fake image)
        noise = Variable(torch.randn(torch_input.size()[0], 100, 1, 1))
        fake = netG(noise)
        target = Variable(torch.zeros(torch_input.size()[0]))
        output = netD(fake.detach())
        errD_fake = criterion(output, target)

        # Backpropagation (discriminator)
        errD = errD_real + errD_fake
        errD.backward()
        optimiserD.step()

        # Train generator

        # Initialise grad with
        # respect to the weights
        # to 0
        netG.zero_grad()
        target = Variable(torch.ones(torch_input.size()[0]))
        output = netD(fake)
        errG = criterion(output, target)

        # Backpropagation (generator)
        errG.backward()
        optimiserG.step()

        # Logging
        print("[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f" % (epoch, 25, i, len(dataloader), errD.data[0], errG.data[0]))

        # Save images every 100 steps
        if i % 100 == 0:
            vutils.save_image(real, "%s/real_samples.png" % "./results", normalize = True)
            fake = netG(noise)
            vutils.save_image(fake.data, "%s/fake_samples_epoch_%03d.png" % ("./results", epoch), normalize = True)
