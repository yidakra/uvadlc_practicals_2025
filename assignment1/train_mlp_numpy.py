################################################################################
# MIT License
#
# Copyright (c) 2025 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2025
# Date Created: 2025-10-28
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from tqdm.auto import tqdm
from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils

import torch


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions between 0 and 1,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Get predicted class (argmax over class dimension)
    predicted_classes = np.argmax(predictions, axis=1)

    # Handle both 1D class indices and 2D one-hot encoded targets
    if targets.ndim == 2:
        # One-hot encoded, convert to class indices
        target_classes = np.argmax(targets, axis=1)
    else:
        target_classes = targets

    # Compute accuracy
    accuracy = np.mean(predicted_classes == target_classes)
    #######################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def evaluate_model(model, data_loader):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
      avg_accuracy: scalar float, the average accuracy of the model on the dataset.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    total_correct = 0
    total_samples = 0

    for data, targets in data_loader:
        # Flatten images: (batch_size, C, H, W) -> (batch_size, C*H*W)
        batch_size = data.shape[0]
        data = data.reshape(batch_size, -1)

        # Forward pass
        predictions = model.forward(data)

        # Compute accuracy for this batch
        predicted_classes = np.argmax(predictions, axis=1)
        if targets.ndim == 2:
            target_classes = np.argmax(targets, axis=1)
        else:
            target_classes = targets

        total_correct += np.sum(predicted_classes == target_classes)
        total_samples += batch_size

    avg_accuracy = total_correct / total_samples
    #######################
    # END OF YOUR CODE    #
    #######################

    return avg_accuracy


def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation. Between 0.0 and 1.0
      logging_dict: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model. 
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set, 
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    ## Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=True)

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # TODO: Initialize model and loss module
    # CIFAR-10 images are 3x32x32 = 3072 pixels, 10 classes
    n_inputs = 3 * 32 * 32
    n_classes = 10
    model = MLP(n_inputs, hidden_dims, n_classes)
    loss_module = CrossEntropyModule()

    # TODO: Training loop including validation
    val_accuracies = []
    train_losses = []
    best_val_accuracy = 0
    best_model = None

    for epoch in range(epochs):
        # Training phase
        epoch_losses = []

        for data, targets in tqdm(cifar10_loader['train'], desc=f'Epoch {epoch+1}/{epochs}', leave=False):
            # Flatten images: (batch_size, C, H, W) -> (batch_size, C*H*W)
            batch_size_curr = data.shape[0]
            data = data.reshape(batch_size_curr, -1)

            # Forward pass
            predictions = model.forward(data)

            # Compute loss
            loss = loss_module.forward(predictions, targets)
            epoch_losses.append(loss)

            # Backward pass
            dout = loss_module.backward(predictions, targets)
            model.backward(dout)

            # Update parameters using SGD
            for module in model.modules:
                if hasattr(module, 'params'):
                    for param_name in module.params:
                        module.params[param_name] -= lr * module.grads[param_name]

        # Validation phase
        val_accuracy = evaluate_model(model, cifar10_loader['validation'])
        val_accuracies.append(val_accuracy)

        avg_train_loss = np.mean(epoch_losses)
        train_losses.append(avg_train_loss)

        print(f'Epoch {epoch+1}/{epochs}: Train Loss = {avg_train_loss:.4f}, Val Accuracy = {val_accuracy:.4f}')

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = deepcopy(model)

    # TODO: Test best model
    test_accuracy = evaluate_model(best_model, cifar10_loader['test'])
    print(f'Test Accuracy: {test_accuracy:.4f}')

    # TODO: Add any information you might want to save for plotting
    logging_dict = {
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'test_accuracy': test_accuracy
    }
    #######################
    # END OF YOUR CODE    #
    #######################

    return best_model, val_accuracies, test_accuracy, logging_dict


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here
    