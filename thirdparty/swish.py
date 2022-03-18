# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from a file in the following repo
# (released under the Apache License 2.0).
#
# Source:
# https://github.com/ceshine/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/utils.py
#
# The license for the original version of this file can be
# found in this directory (LICENSE_apache).
# ---------------------------------------------------------------

import torch
import torch.nn as nn


class SwishSN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


def act(t):
    # The following implementation has lower memory.
    return SwishSN.apply(t)


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return act(x)