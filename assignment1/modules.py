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
        # using kaiming initialization for elu layers
        # relu and elu use std = sqrt(2 / fan_in)
        std = np.sqrt(2.0 / in_features)
        self.params['weight'] = np.random.normal(0, std, (out_features, in_features))
        self.params['bias'] = np.zeros((1, out_features))

        # stash zeroed gradients for later updates
        self.grads['weight'] = np.zeros((out_features, in_features))
        self.grads['bias'] = np.zeros((1, out_features))

        # keep input for the backward pass
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
        # remember the input for backward
        self.cache = x

        # compute the linear response
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
        x = self.cache

        # accumulate gradients for weights and bias
        self.grads['weight'] = dout.T @ x

        # sum over the batch for the bias gradient
        self.grads['bias'] = np.sum(dout, axis=0, keepdims=True)

        # propagate the gradient downstream
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
        # compute elu with its positive and negative halves
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
        x = self.cache
        grad = np.where(x > 0, 1.0, self.alpha * np.exp(x))

        # apply the chain rule elementwise
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
        # use the max trick to keep the exponentials stable
        x_max = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - x_max)
        out = exp_x / np.sum(exp_x, axis=1, keepdims=True)

        # cache the probabilities for backward
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
        y = self.cache
        # compute the dot between upstream grad and probabilities per sample
        sum_term = np.sum(dout * y, axis=1, keepdims=True)
        # pull gradients back through the softmax
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
        batch_size = x.shape[0]

        # support class indices or one-hot targets
        if y.ndim == 1:
            n_classes = x.shape[1]
            y_one_hot = np.zeros_like(x)
            y_one_hot[np.arange(batch_size), y] = 1
            y = y_one_hot

        # guard against log(0) with a small epsilon
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
        batch_size = x.shape[0]

        # support class indices or one-hot targets
        if y.ndim == 1:
            n_classes = x.shape[1]
            y_one_hot = np.zeros_like(x)
            y_one_hot[np.arange(batch_size), y] = 1
            y = y_one_hot

        # compute the gradient of the loss
        epsilon = 1e-12
        dx = -y / (x + epsilon) / batch_size
        #######################
        # END OF YOUR CODE    #
        #######################

        return dx