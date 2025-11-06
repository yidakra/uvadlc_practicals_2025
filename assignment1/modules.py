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
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features, input_layer=False):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample
          input_layer: boolean, True if this is the first layer after the input, else False.

        TODO:
        Initialize weight parameters using Kaiming initialization. 
        Initialize biases with zeros.
        Hint: the input_layer argument might be needed for the initialization

        Also, initialize gradients with zeros.
        """

        # Note: For the sake of this assignment, please store the parameters
        # and gradients in this format, otherwise some unit tests might fail.
        self.params = {'weight': None, 'bias': None} # Model parameters
        self.grads = {'weight': None, 'bias': None} # Gradients

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # Kaiming initialization for weights
        # For ReLU/ELU activations, use std = sqrt(2/fan_in)
        std = np.sqrt(2.0 / in_features)
        self.params['weight'] = np.random.normal(0, std, (out_features, in_features))
        self.params['bias'] = np.zeros((1, out_features))

        # Initialize gradients with zeros
        self.grads['weight'] = np.zeros((out_features, in_features))
        self.grads['bias'] = np.zeros((1, out_features))

        # Cache for backward pass
        self.cache = None
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # Forward pass: Y = XW^T + B
        # Store input for backward pass
        self.cache = x

        # Compute output: out = x @ W^T + b
        out = x @ self.params['weight'].T + self.params['bias']
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # Backward pass
        # dout has shape (batch_size, out_features)
        # From the PDF Question 1:
        # dL/dW = (dL/dY)^T @ X, shape (out_features, in_features)
        # dL/db = sum over batch of dL/dY, shape (1, out_features)
        # dL/dX = dL/dY @ W, shape (batch_size, in_features)

        x = self.cache

        # Gradient w.r.t. weights: dL/dW = dout^T @ x
        self.grads['weight'] = dout.T @ x

        # Gradient w.r.t. bias: sum over batch dimension
        self.grads['bias'] = np.sum(dout, axis=0, keepdims=True)

        # Gradient w.r.t. input: dL/dX = dout @ W
        dx = dout @ self.params['weight']
        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.cache = None
        #######################
        # END OF YOUR CODE    #
        #######################


class ELUModule(object):
    """
    ELU activation module.
    """

    def __init__(self, alpha):
        self.alpha = alpha
        self.cache = None

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # ELU(x) = x if x > 0, else alpha * (exp(x) - 1)
        self.cache = x
        out = np.where(x > 0, x, self.alpha * (np.exp(x) - 1))
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # Derivative of ELU:
        # dELU/dx = 1 if x > 0
        # dELU/dx = alpha * exp(x) if x <= 0
        x = self.cache
        grad = np.where(x > 0, 1.0, self.alpha * np.exp(x))

        # Apply chain rule with Hadamard product (element-wise multiplication)
        dx = dout * grad
        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.cache = None
        #######################
        # END OF YOUR CODE    #
        #######################


class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def __init__(self):
        self.cache = None

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # Softmax with max trick for numerical stability
        # Y_ij = exp(X_ij - max_k(X_ik)) / sum_k(exp(X_ik - max_k(X_ik)))
        x_max = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - x_max)
        out = exp_x / np.sum(exp_x, axis=1, keepdims=True)

        # Store output for backward pass
        self.cache = out
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # Backward pass for softmax
        # For each sample i and class j:
        # dL/dX_ij = sum_k (dL/dY_ik * dY_ik/dX_ij)
        #          = sum_k (dL/dY_ik * Y_ik * (δ_jk - Y_ij))
        #          = Y_ij * (dL/dY_ij - sum_k(dL/dY_ik * Y_ik))

        y = self.cache
        # Compute sum_k(dout_ik * y_ik) for each sample
        sum_term = np.sum(dout * y, axis=1, keepdims=True)
        # Apply the formula: dx = y * (dout - sum_term)
        dx = y * (dout - sum_term)
        #######################
        # END OF YOUR CODE    #
        #######################

        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.cache = None
        #######################
        # END OF YOUR CODE    #
        #######################


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module (predicted probabilities from softmax)
          y: labels of the input (one-hot encoded or class indices)
        Returns:
          out: cross entropy loss

        TODO:
        Implement forward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # Cross entropy loss: L = -1/S * sum_i,k (T_ik * log(Y_ik))
        # x is the predicted probabilities (after softmax)
        # y can be class indices or one-hot encoded

        batch_size = x.shape[0]

        # Handle both class indices and one-hot encoded labels
        if y.ndim == 1:
            # y is class indices, convert to one-hot
            n_classes = x.shape[1]
            y_one_hot = np.zeros_like(x)
            y_one_hot[np.arange(batch_size), y] = 1
            y = y_one_hot

        # Compute cross entropy loss with small epsilon for numerical stability
        epsilon = 1e-12
        out = -np.mean(np.sum(y * np.log(x + epsilon), axis=1))
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module (predicted probabilities from softmax)
          y: labels of the input (one-hot encoded or class indices)
        Returns:
          dx: gradient of the loss with the respect to the input x.

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # Backward pass: dL/dY = -1/S * T / Y (element-wise division)
        batch_size = x.shape[0]

        # Handle both class indices and one-hot encoded labels
        if y.ndim == 1:
            # y is class indices, convert to one-hot
            n_classes = x.shape[1]
            y_one_hot = np.zeros_like(x)
            y_one_hot[np.arange(batch_size), y] = 1
            y = y_one_hot

        # Gradient: dL/dx = -1/S * y / x
        epsilon = 1e-12
        dx = -y / (x + epsilon) / batch_size
        #######################
        # END OF YOUR CODE    #
        #######################

        return dx