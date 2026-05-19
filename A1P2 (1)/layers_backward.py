"""THWS/MAI/Introduction to Deep Learning - Assignment 2 Part 2 - layers with backward pass

Magda Gregorová, April 2026

This file extends the classes you implemented in Part 1.

Instructions:
  1. Copy your Linear, ReLU, and Model classes from layers.py (Part 1) into this file.
  2. Add the backward methods to Linear, ReLU, and Model as described below.
  3. Do not modify the forward methods.

Use only PyTorch tensor operations — no numpy, no torch.nn.
"""

import torch
from functions import linear_ggrad, relu_ggrad


# TODO: Copy your Linear class from Part 1 here and add the backward method below.
#
# class Linear:
#
#     def __init__(self, theta_1, theta_0):
#         ...  # your Part 1 implementation
#
#     def forward(self, ins):
#         ...  # your Part 1 implementation
#
#     def backward(self, gout):
#         """Backward pass: compute and store global gradients.
#
#         Stores gradients as attributes on the tensors:
#             self.theta_1.g of shape (out_features, in_features)
#             self.theta_0.g of shape (1, out_features)
#             self.ins.g     of shape (N, in_features)
#
#         Args:
#             gout: torch.tensor of shape (N, out_features) - upstream gradient dL/dZ
#
#         Returns:
#             torch.tensor of shape (N, in_features) - gradient w.r.t. input
#         """
#         # TODO: assign results to .g attributes, return self.ins.g
#         pass


# TODO: Copy your ReLU class from Part 1 here and add the backward method below.
#
# class ReLU:
#
#     def forward(self, ins):
#         ...  # your Part 1 implementation
#
#     def backward(self, gout):
#         """Backward pass: compute and store global gradient.
#
#         Stores gradient as attribute on the tensor:
#             self.ins.g of same shape as self.ins
#
#         Args:
#             gout: torch.tensor of same shape as self.ins - upstream gradient dL/dA
#
#         Returns:
#             torch.tensor of same shape - gradient w.r.t. input
#         """
#         # TODO: assign result to self.ins.g, return self.ins.g
#         pass


# TODO: Copy your Model class from Part 1 here and add the backward method below.
#
# class Model:
#
#     def __init__(self, layers):
#         ...  # your Part 1 implementation
#
#     def forward(self, ins):
#         ...  # your Part 1 implementation
#
#     def backward(self, gout):
#         """Backward pass through all layers in reverse order.
#
#         Args:
#             gout: torch.tensor - upstream gradient from the loss
#
#         Returns:
#             torch.tensor - gradient w.r.t. network input
#         """
#         # TODO: implement
#         pass
