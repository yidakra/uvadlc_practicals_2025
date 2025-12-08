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
from torchvision.utils import make_grid
import numpy as np


def sample_reparameterize(mean, std):
    """
    Perform the reparameterization trick to sample from a distribution with the given mean and std
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions
        std - Tensor of arbitrary shape with strictly positive values. Denotes the standard deviation
              of the distribution
    Outputs:
        z - A sample of the distributions, with gradient support for both mean and std.
            The tensor should have the same shape as the mean and std input tensors.
    """
    assert not (std < 0).any().item(), "The reparameterization trick got a negative std as input. " + \
                                       "Are you sure your input is std and not log_std?"
    # Sample epsilon from standard normal distribution
    epsilon = torch.randn_like(std)

    # Reparameterization: z = mean + std * epsilon
    z = mean + std * epsilon
    return z


def KLD(mean, log_std):
    """
    Calculates the Kullback-Leibler divergence of given distributions to unit Gaussians over the last dimension.
    See the definition of the regularization loss in Section 1.4 for the formula.
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions.
        log_std - Tensor of arbitrary shape and range, denoting the log standard deviation of the distributions.
    Outputs:
        KLD - Tensor with one less dimension than mean and log_std (summed over last dimension).
              The values represent the Kullback-Leibler divergence to unit Gaussians.
    """

    # Compute KL divergence using the closed form formula from Equation 13
    # KL(N(mu, sigma^2) || N(0, 1)) = 0.5 * sum(exp(2*log_sigma) + mu^2 - 1 - 2*log_sigma)
    KLD = 0.5 * torch.sum(torch.exp(2 * log_std) + mean**2 - 1 - 2 * log_std, dim=-1)
    return KLD


def elbo_to_bpd(elbo, img_shape):
    """
    Converts the summed negative log likelihood given by the ELBO into the bits per dimension score.
    Inputs:
        elbo - Tensor of shape [batch_size]
        img_shape - Shape of the input images, representing [batch, channels, height, width]
    Outputs:
        bpd - The negative log likelihood in bits per dimension for the given image.
    """
    # Calculate total number of dimensions per image (excluding batch dimension)
    # For MNIST: channels * height * width = 1 * 28 * 28 = 784
    num_dims = np.prod(img_shape[1:])

    # Convert nats to bits: multiply by log2(e)
    # Normalize by number of dimensions
    bpd = elbo * np.log2(np.e) / num_dims
    return bpd


@torch.no_grad()
def visualize_manifold(decoder, grid_size=20):
    """
    Visualize a manifold over a 2 dimensional latent space. The images in the manifold
    should represent the decoder's output means (not binarized samples of those).
    Inputs:
        decoder - Decoder model such as LinearDecoder or ConvolutionalDecoder.
        grid_size - Number of steps/images to have per axis in the manifold.
                    Overall you need to generate grid_size**2 images, and the distance
                    between different latents in percentiles is 1/grid_size
    Outputs:
        img_grid - Grid of images representing the manifold.
    """

    ## Hints:
    # - You can use the icdf method of the torch normal distribution  to obtain z values at percentiles.
    # - Use the range [0.5/grid_size, 1.5/grid_size, ..., (grid_size-0.5)/grid_size] for the percentiles.
    # - torch.meshgrid might be helpful for creating the grid of values
    # - You can use torchvision's function "make_grid" to combine the grid_size**2 images into a grid
    # - Remember to apply a softmax after the decoder

    # Create percentile values for grid_size points
    # Range: [0.5/grid_size, 1.5/grid_size, ..., (grid_size-0.5)/grid_size]
    percentiles = torch.linspace(0.5 / grid_size, 1 - 0.5 / grid_size, grid_size)

    # Use inverse CDF (ppf) of standard normal to get z values at these percentiles
    normal_dist = torch.distributions.Normal(0, 1)
    z_values = normal_dist.icdf(percentiles)

    # Create 2D grid of z values
    z1_grid, z2_grid = torch.meshgrid(z_values, z_values, indexing='ij')

    # Flatten and stack to create batch of latent vectors [grid_size^2, 2]
    z_grid = torch.stack([z1_grid.flatten(), z2_grid.flatten()], dim=1).to(decoder.device)

    # Decode the latent vectors to get reconstructions
    decoder_output = decoder(z_grid)  # [grid_size^2, 16, 28, 28]

    # Apply softmax to get probabilities and take the mean (expected value)
    # For categorical distribution, the mean is sum of k * p_k
    probs = torch.softmax(decoder_output, dim=1)  # [grid_size^2, 16, 28, 28]

    # Calculate expected pixel values: sum over categories weighted by probabilities
    # Categories are 0-15, so we weight by k
    pixel_values = torch.sum(probs * torch.arange(16, device=decoder.device).view(1, -1, 1, 1), dim=1, keepdim=True)

    # Normalize to [0, 1] range for visualization
    pixel_values = pixel_values / 15.0

    # Create grid of images
    img_grid = make_grid(pixel_values, nrow=grid_size, normalize=False, pad_value=0.5)
    return img_grid

