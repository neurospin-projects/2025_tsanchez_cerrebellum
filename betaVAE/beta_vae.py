# -*- coding: utf-8 -*-

# /usr/bin/env python3
#
#  This software and supporting documentation are distributed by
#      Institut Federatif de Recherche 49
#      CEA/NeuroSpin, Batiment 145,
#      91191 Gif-sur-Yvette cedex
#      France
#
# This software is governed by the CeCILL license version 2 under
# French law and abiding by the rules of distribution of free software.
# You can  use, modify and/or redistribute the software under the
# terms of the CeCILL license version 2 as circulated by CEA, CNRS
# and INRIA at the following URL "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license version 2 and that you accept its terms.

from collections import OrderedDict
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from omegaconf import DictConfig
from betaVAE.backbones.basicnet import BasicNet
from betaVAE.backbones.convnet import ConvNet
from betaVAE.backbones.resnet import ResNet, BasicBlock



class VAE(nn.Module):
    """ beta-VAE class
    """
    def __init__(self, config : DictConfig, device = None):
        """
        Args:
            in_shape: tuple, input shape
            n_latent: int, latent space size
            depth: int, depth of the model
        """
        super().__init__()

        if device :
            self.device = device
        else : 
            if torch.cuda.is_available() : 
                self.device = "cuda" #Using default GPU
            else : 
                self.device = "cpu"

        self.n_latent = config.n
        
        match config.backbone_name:
            case "basicnet" :
                self.encoder = BasicNet(
                    in_shape=config.in_shape,
                    depth = config.encoder_depth,
                )

                self.out_dim = self.encoder.z_dim_h * self.encoder.z_dim_w * self.encoder.z_dim_d
                self.final_nb_filters = 16 * (2 ** (config.encoder_depth -1))
                self.flatten_size = self.final_nb_filters * self.out_dim
                
            case "convnet" : 
                self.encoder = ConvNet(
                    encoder_depth=config.encoder_depth,
                    filters=config.filters,
                    block_depth=config.block_depth,
                    initial_kernel_size=config.initial_kernel_size,
                    initial_stride=config.initial_stride,
                    max_pool=config.max_pool,
                    num_representation_features=config.backbone_output_size,
                    linear = config.linear_in_backbone,
                    adaptive_pooling=config.adaptive_pooling,
                    drop_rate=config.drop_rate,
                    in_shape=config.in_shape)

                self.out_dim = self.encoder.out_dim 
                self.final_nb_filters = config.filters[-1]
                self.flatten_size = self.out_dim * self.final_nb_filters

            case "resnet" :
                self.encoder = ResNet(
                    block=BasicBlock,
                    layers=config.layers,
                    channels=config.channels,
                    in_channels=1,
                    num_classes=config.backbone_output_size,
                    zero_init_residual=config.zero_init_residual,
                    dropout_rate=config.drop_rate,
                    out_block=None,
                    prediction_bias=False,
                    initial_kernel_size=config.initial_kernel_size,
                    initial_stride=config.initial_stride,
                    adaptive_pooling=config.adaptive_pooling,
                    linear_in_backbone=config.linear_in_backbone)
                

        # * Mean and var computation
        self.z_mean = nn.Linear(self.flatten_size, self.n_latent)
        self.z_var = nn.Linear(self.flatten_size, self.n_latent)
        self.z_develop = nn.Linear(self.n_latent, self.flatten_size)
        
        #* Same decoder for the different architectures
        modules_decoder = []
        for step in range(config.encoder_depth-1):
            in_channels = out_channels if step != 0 else self.final_nb_filters
            out_channels = in_channels // 2
            ini = 1 if step==0 else 0
            modules_decoder.append(('convTrans3d%s' %step, nn.ConvTranspose3d(in_channels,
                        out_channels, kernel_size=2, stride=2, padding=0, output_padding=(ini,0,0))))
            modules_decoder.append(('normup%s' %step, nn.BatchNorm3d(out_channels)))
            modules_decoder.append(('ReLU%s' %step, nn.ReLU()))
            modules_decoder.append(('convTrans3d%sa' %step, nn.ConvTranspose3d(out_channels,
                        out_channels, kernel_size=3, stride=1, padding=1)))
            modules_decoder.append(('normup%sa' %step, nn.BatchNorm3d(out_channels)))
            modules_decoder.append(('ReLU%sa' %step, nn.ReLU()))
        modules_decoder.append(('convtrans3dn', nn.ConvTranspose3d(16, 1, kernel_size=2,
                        stride=2, padding=0)))
        modules_decoder.append(('conv_final', nn.Conv3d(1, 2, kernel_size=1, stride=1)))
        self.decoder = nn.Sequential(OrderedDict(modules_decoder))
        
        # ! Check weight initialisation for the different layers
        self.weight_initialization()

    def weight_initialization(self):
        """
        Initializes model parameters according to Gaussian Glorot initialization
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.ConvTranspose3d) or isinstance(module, nn.Conv3d):
                if "conv_final" in name:
                    nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                    nn.init.constant_(module.bias, 0)
                else:
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def sample_z(self, mean, logvar):
        device = self.device
        stddev = torch.exp(0.5 * logvar)
        noise = Variable(torch.randn(stddev.size(), device=device))
        return (noise * stddev) + mean

    def encode(self, x, return_feat = False):
        feat = self.encoder(x)
        x = nn.functional.normalize(feat, p=2)
        x = x.view(x.size(0), -1)
        mean = self.z_mean(x)
        var = self.z_var(x)
        if return_feat : 
            return mean, var, feat
        else : 
            return mean, var

    def decode(self, z):
        out = self.z_develop(z)
        out = out.view(z.size(0), self.final_nb_filters, self.encoder.z_dim_h, self.encoder.z_dim_w, self.encoder.z_dim_d)
        out = self.decoder(out)
        return out

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.sample_z(mean, logvar)
        out = self.decode(z)
        return out, mean, logvar


def vae_loss(output, inputs, mean, logvar, loss_func, kl_weight):
    kl_loss = -0.5 * torch.sum(-torch.exp(logvar) - mean**2 + 1. + logvar)
    recon_loss = loss_func(output, inputs)
    return recon_loss, kl_loss, recon_loss + (kl_weight * kl_loss)
