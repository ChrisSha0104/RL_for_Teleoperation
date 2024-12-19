import math
import abc
import numpy as np
import textwrap
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models as vision_models

class Module(torch.nn.Module):
    """
    Base class for networks. The only difference from torch.nn.Module is that it
    requires implementing @output_shape.
    """
    @abc.abstractmethod
    def output_shape(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        raise NotImplementedError

class ConvBase(Module):
    """
    Base class for ConvNets.
    """
    def __init__(self):
        super(ConvBase, self).__init__()

    # dirty hack - re-implement to pass the buck onto subclasses from ABC parent
    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        raise NotImplementedError

    def forward(self, inputs):
        x = self.nets(inputs)
        if list(self.output_shape(list(inputs.shape)[1:])) != list(x.shape)[1:]:
            raise ValueError('Size mismatch: expect size %s, but got size %s' % (
                str(self.output_shape(list(inputs.shape)[1:])), str(list(x.shape)[1:]))
            )
        return x

class ResNet18Conv(ConvBase):
    """
    A ResNet18 block that can be used to process input images.
    """
    def __init__(
        self,
        input_channel=3,
        pretrained=False,
        input_coord_conv=False,
        mlp_input_dim=512*3*3,
        mlp_output_dim=128,
    ):
        """
        Args:
            input_channel (int): number of input channels for input images to the network.
                If not equal to 3, modifies first conv layer in ResNet to handle the number
                of input channels.
            pretrained (bool): if True, load pretrained weights for all ResNet layers.
            input_coord_conv (bool): if True, use a coordinate convolution for the first layer
                (a convolution where input channels are modified to encode spatial pixel location)
        """
        super(ResNet18Conv, self).__init__()
        net = vision_models.resnet18(pretrained=pretrained)

        if input_coord_conv:
            net.conv1 = CoordConv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif input_channel != 3:
            net.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # cut the last fc layer
        self._input_coord_conv = input_coord_conv
        self._input_channel = input_channel
        self.nets = torch.nn.Sequential(*(list(net.children())[:-2]))
        self.mlp_output_dim = mlp_output_dim

        # # Define an MLP to reduce dimensions
        self.mlp = nn.Sequential(
            nn.Flatten(),  # Flatten the spatial dimensions (B, C, H, W) → (B, C*H*W)
            nn.Linear(512*3*3, 1024),
            nn.ReLU(),  # Activation
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512, mlp_output_dim)  # Reduce to desired output dimension
        )

    def forward(self, x): 
        """
        Forward pass through the encoder and the MLP.
        """
        features = self.nets(x)
        reduced_features = self.mlp(features)
        return reduced_features


    # def output_shape(self, input_shape):
    #     """
    #     Function to compute output shape from inputs to this module. 

    #     Args:
    #         input_shape (iterable of int): shape of input. Does not include batch dimension.
    #             Some modules may not need this argument, if their output does not depend 
    #             on the size of the input, or if they assume fixed size input.

    #     Returns:
    #         out_shape ([int]): list of integers corresponding to output shape
    #     """
    #     assert(len(input_shape) == 3)
    #     out_h = int(math.ceil(input_shape[1] / 32.))
    #     out_w = int(math.ceil(input_shape[2] / 32.))
    #     return [512, out_h, out_w]

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        return header + '(input_channel={}, input_coord_conv={})'.format(self._input_channel, self._input_coord_conv)
