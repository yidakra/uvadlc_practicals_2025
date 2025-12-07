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
    Sample from a Gaussian distribution using the reparameterization trick.
    
    Parameters:
        mean (Tensor): Mean of the distribution; can have any shape.
        std (Tensor): Standard deviation, broadcastable to `mean`. Must not contain negative values.
    
    Returns:
        z (Tensor): A sample with the same shape as `mean` and `std`. Gradients propagate to `mean` and `std`.
    """
    assert not (std < 0).any().item(), "The reparameterization trick got a negative std as input. " + \
                                       "Are you sure your input is std and not log_std?"
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Sample epsilon from standard normal distribution
    epsilon = torch.randn_like(std)

    # Reparameterization: z = mean + std * epsilon
    z = mean + std * epsilon
    #######################
    # END OF YOUR CODE    #
    #######################
    return z


def KLD(mean, log_std):
    """
    Compute the Kullback–Leibler divergence from N(mean, exp(2*log_std)) to the standard normal, summed over the last dimension.
    
    Parameters:
        mean: Tensor of means for each Gaussian.
        log_std: Tensor of log standard deviations (natural log of the standard deviation), broadcastable to mean.
    
    Returns:
        Tensor of KLD values with the last dimension reduced (shape equal to mean.shape[:-1]). Each value equals
        0.5 * sum(exp(2 * log_std) + mean**2 - 1 - 2 * log_std) taken over the last dimension.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Compute KL divergence using the closed form formula from Equation 13
    # KL(N(mu, sigma^2) || N(0, 1)) = 0.5 * sum(exp(2*log_sigma) + mu^2 - 1 - 2*log_sigma)
    KLD = 0.5 * torch.sum(torch.exp(2 * log_std) + mean**2 - 1 - 2 * log_std, dim=-1)
    #######################
    # END OF YOUR CODE    #
    #######################
    return KLD


def elbo_to_bpd(elbo, img_shape):
    """
    Convert summed negative log-likelihood (ELBO) per example into bits-per-dimension for a given image shape.
    
    Parameters:
        elbo (Tensor): Negative log-likelihood per example, shape [batch_size], expressed in nats.
        img_shape (Sequence[int]): Image tensor shape [batch, channels, height, width] used to determine the number of dimensions per example.
    
    Returns:
        Tensor: Bits-per-dimension (bpd) for each example, shape [batch_size]; computed by converting nats to bits and dividing by the number of image dimensions (channels * height * width).
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Calculate total number of dimensions per image (excluding batch dimension)
    # For MNIST: channels * height * width = 1 * 28 * 28 = 784
    num_dims = np.prod(img_shape[1:])

    # Convert nats to bits: multiply by log2(e)
    # Normalize by number of dimensions
    bpd = elbo * np.log2(np.e) / num_dims
    #######################
    # END OF YOUR CODE    #
    #######################
    return bpd


@torch.no_grad()
def visualize_manifold(decoder, grid_size=20):
    """
    Builds an image grid visualizing the decoder's output means across a 2-dimensional latent manifold.
    
    The decoder is sampled on a grid of latent-percentiles mapped through the standard normal inverse CDF; decoder outputs are interpreted as categorical logits per pixel (expected pixel value computed from channel probabilities), scaled to [0, 1], and arranged into a grid image.
    
    Parameters:
        decoder: A decoder callable that accepts a batch of latent vectors with shape [N, 2] and returns logits with shape [N, K, H, W] (the implementation assumes K == 16). The decoder must expose a `.device` attribute used for tensor placement.
        grid_size (int): Number of points per axis in the latent grid; total images produced = grid_size**2.
    
    Returns:
        img_grid (torch.Tensor): A single image tensor containing the grid of decoded means, suitable for visualization (shape [C, H_img, W_img]).
    """

    ## Hints:
    # - You can use the icdf method of the torch normal distribution  to obtain z values at percentiles.
    # - Use the range [0.5/grid_size, 1.5/grid_size, ..., (grid_size-0.5)/grid_size] for the percentiles.
    # - torch.meshgrid might be helpful for creating the grid of values
    # - You can use torchvision's function "make_grid" to combine the grid_size**2 images into a grid
    # - Remember to apply a softmax after the decoder

    #######################
    # PUT YOUR CODE HERE  #
    #######################
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
    #######################
    # END OF YOUR CODE    #
    #######################

    return img_grid
