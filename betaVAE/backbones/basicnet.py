"""Network that Louise used initially for the reconstructions"""

import torch.nn as nn
from collections import OrderedDict
from typing import List

class BasicNet(nn.Module):
    def __init__(self, in_shape : List[int], depth = 3):

        super().__init__()
        self.in_shape = in_shape
        c,h,w,d = in_shape
        self.depth = depth
        self.z_dim_h = h//2**depth # receptive field downsampled 2 times for each step
        self.z_dim_w = w//2**depth
        self.z_dim_d = d//2**depth

        modules_encoder = []
        for step in range(depth):
            in_channels = 1 if step == 0 else out_channels
            out_channels = 16 if step == 0  else 16 * (2**step)
            modules_encoder.append(('conv%s' %step, nn.Conv3d(in_channels, out_channels,
                    kernel_size=3, stride=1, padding=1)))
            modules_encoder.append(('norm%s' %step, nn.BatchNorm3d(out_channels)))
            modules_encoder.append(('LeakyReLU%s' %step, nn.LeakyReLU()))
            modules_encoder.append(('conv%sa' %step, nn.Conv3d(out_channels, out_channels,
                    kernel_size=4, stride=2, padding=1)))
            modules_encoder.append(('norm%sa' %step, nn.BatchNorm3d(out_channels)))
            modules_encoder.append(('LeakyReLU%sa' %step, nn.LeakyReLU()))
        self.encoder = nn.Sequential(OrderedDict(modules_encoder))

    def forward(self, x):
        x_hat = self.encoder(x)
        return x_hat