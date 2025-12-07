################################################################################
# MIT License
#
# Copyright (c) 2022
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Autumn 2022
# Date Created: 2022-11-25
################################################################################

import torch
import torch.nn as nn
import numpy as np


class CNNEncoder(nn.Module):
    def __init__(self, num_input_channels: int = 1, num_filters: int = 32,
                 z_dim: int = 20):
        """
                 Create a CNN-based encoder that maps input images to parameters of a latent distribution.
                 
                 Parameters:
                     num_input_channels (int): Number of image channels (e.g., 1 for MNIST).
                     num_filters (int): Number of filters in the first convolutional layer; deeper layers use multiples of this.
                     z_dim (int): Dimensionality of the latent representation produced for mean and log standard deviation.
                 """
        # For an intial architecture, you can use the encoder of Tutorial 9.
        # Feel free to experiment with the architecture yourself, but the one specified here is
        # sufficient for the assignment.
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        super().__init__()

        # CNN architecture similar to Tutorial 9
        # Encodes 28x28 images through convolutional layers
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, num_filters, kernel_size=3, padding=1, stride=2),  # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_filters, 2*num_filters, kernel_size=3, padding=1, stride=2),  # 14x14 -> 7x7
            nn.ReLU(),
            nn.Conv2d(2*num_filters, 2*num_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(2*num_filters, 2*num_filters, kernel_size=3, padding=1, stride=2),  # 7x7 -> 4x4
            nn.ReLU(),
            nn.Flatten(),  # flatten to 2*num_filters*4*4
        )

        # Linear layers to output mean and log_std for latent distribution
        self.fc_mean = nn.Linear(2*num_filters*4*4, z_dim)
        self.fc_log_std = nn.Linear(2*num_filters*4*4, z_dim)
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Encode a batch of images into latent distribution parameters.
        
        Parameters:
            x (torch.Tensor): Input batch with shape [B, C, H, W], integer values in {0,…,15}. Inputs are cast to float and normalized to the range [-1, 1].
        
        Returns:
            mean (torch.Tensor): Tensor of shape [B, z_dim] containing the predicted latent means.
            log_std (torch.Tensor): Tensor of shape [B, z_dim] containing the predicted latent log standard deviations.
        """
        x = x.float() / 15 * 2.0 - 1.0  # Move images between -1 and 1
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # Pass through convolutional layers
        h = self.net(x)

        # Output mean and log_std for the latent distribution
        mean = self.fc_mean(h)
        log_std = self.fc_log_std(h)
        #######################
        # END OF YOUR CODE    #
        #######################
        return mean, log_std


class CNNDecoder(nn.Module):
    def __init__(self, num_input_channels: int = 16, num_filters: int = 32,
                 z_dim: int = 20):
        """
                 Create a CNN decoder that maps latent vectors to image logits.
                 
                 Parameters:
                     num_input_channels (int): Number of output channels in reconstructed images (e.g., 16 for 4-bit MNIST).
                     num_filters (int): Base number of convolutional filters used in decoder feature maps.
                     z_dim (int): Dimensionality of the latent representation to be decoded.
                 """
        # For an intial architecture, you can use the decoder of Tutorial 9.
        # Feel free to experiment with the architecture yourself, but the one specified here is
        # sufficient for the assignment.
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        super().__init__()

        # Linear layer to map from latent space to spatial feature map
        self.linear = nn.Sequential(
            nn.Linear(z_dim, 2*num_filters*4*4),
            nn.ReLU()
        )

        # Transposed convolutions to upsample back to 28x28
        # Mirror the encoder architecture
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2*num_filters, 2*num_filters, kernel_size=3, output_padding=1, padding=1, stride=2),  # 4x4 -> 7x7
            nn.ReLU(),
            nn.Conv2d(2*num_filters, 2*num_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(2*num_filters, num_filters, kernel_size=3, output_padding=1, padding=1, stride=2),  # 7x7 -> 14x14
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(num_filters, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2),  # 14x14 -> 28x28
        )
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, z):
        """
        Decode a batch of latent vectors into reconstructed image logits.
        
        Parameters:
            z (torch.Tensor): Latent vectors with shape [B, z_dim].
        
        Returns:
            torch.Tensor: Reconstructed image logits with shape [B, num_input_channels, 28, 28]; raw logits (no softmax or sigmoid applied).
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # Map latent vector to feature map
        x = self.linear(z)
        x = x.reshape(z.shape[0], -1, 4, 4)  # reshape to [B, 2*num_filters, 4, 4]

        # Upsample through transposed convolutions
        x = self.net(x)
        #######################
        # END OF YOUR CODE    #
        #######################
        return x

    @property
    def device(self):
        """
        Property function to get the device on which the decoder is.
        Might be helpful in other functions.
        """
        return next(self.parameters()).device