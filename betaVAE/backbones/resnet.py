"""
Code borrowed from https://github.com/Duplums/SMLvsDL/blob/master/dl_training/models/resnet.py
It belongs to Benoît Dufumier.
"""

import numpy as np
import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if hasattr(self, "concrete_dropout"):
            out = self.concrete_dropout(out)
        else:
            out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Critic(nn.Module):
    """
        Critic used when performing contrastive representation learning
        (inspired from Tsai et al, Conditional Contrastive Learning with Kernel, ICLR 2022)
    """
    def __init__(self, latent_dim):
        super(Critic, self).__init__()
        self.projection_dim = 128
        self.w1 = nn.Linear(latent_dim, latent_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(latent_dim)
        self.relu = nn.ReLU()
        self.w2 = nn.Linear(latent_dim, self.projection_dim, bias=False)
        self.bn2 = nn.BatchNorm1d(self.projection_dim, affine=False)

    def forward(self, x):
        x = self.w1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.w2(x)
        x = self.bn2(x)
        return x


class ResNet(nn.Module):

    def __init__(self, block, layers, channels=[64,128,256,512], in_channels=3, num_classes=1000,
                 zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, dropout_rate=None, out_block=None, prediction_bias=True,
                 initial_kernel_size=7, initial_stride=2, adaptive_pooling=['average', 1], linear_in_backbone=False):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer

        self.name = "resnet"
        self.inputs = None
        self.inplanes = 64
        self.dilation = 1
        self.out_block = out_block
        self.adaptive_pooling = adaptive_pooling
        self.linear_in_backbone = linear_in_backbone

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        #initial_stride = 2 if initial_kernel_size==7 else 1
        padding = (initial_kernel_size-initial_stride+1)//2 # ! k =7, s=1 --> padding = 3
        self.conv1 = nn.Conv3d(in_channels, self.inplanes, kernel_size=initial_kernel_size, stride=initial_stride,
                               padding=padding, bias=False) # * -> No compression in default
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # * channels = [64, 128, 256, 512]
        # * layers = [2,2,2,2]

        self.layer1 = self._make_layer(block, channels[0], layers[0], stride=1)
        self.layer2 = self._make_layer(block, channels[1], layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        # if self.adaptive_pooling[0]=='max':
        #     self.pool = nn.AdaptiveMaxPool3d(self.adaptive_pooling[1])
        # elif self.adaptive_pooling[0]=='average':
        #     self.pool = nn.AdaptiveAvgPool3d(self.adaptive_pooling[1])
        # else:
        #     raise ValueError("Wrong pooling name argument")

        if dropout_rate is not None and dropout_rate>0:
            self.dropout = nn.Dropout(dropout_rate)

        # linear layer to map to embeddings size
        if self.linear_in_backbone:
            output_dim = np.prod(self.adaptive_pooling[1])*channels[-1]
            self.linear = nn.Linear(output_dim, num_classes)

        # attention mechanism
        #self.attention_map = None
        #if out_block is None:
        #    self.fc = nn.Linear(channels[-1] * block.expansion, num_classes, bias=prediction_bias)
        #elif out_block == "contrastive":
        #    self.critic = Critic(channels[-1] * block.expansion)
        #else:
        #    raise NotImplementedError()

        #for m in self.modules():
        #    if isinstance(m, nn.Conv3d):
        #        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #    elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
        #        nn.init.constant_(m.weight, 1)
        #        nn.init.constant_(m.bias, 0)
        #    elif isinstance(m, nn.Linear):
        #        nn.init.normal_(m.weight, 0, 0.01)
        #        if m.bias is not None:
        #            nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
                    

    def get_current_visuals(self):
        return self.inputs

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion # BasicBlock.expansion = 1 ; Bottleneck.expansion = 4
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        self.inputs = x.detach().cpu().numpy()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x) # * Same dim
        #x = self.maxpool(x)

        x1 = self.layer1(x) # * Same dim
        x2 = self.layer2(x1) #* n_in/2
        x3 = self.layer3(x2) #* n_in/2
        x4 = self.layer4(x3) #* n_in/2 

        # x5 = self.pool(x4) # ! Does it make sense to pool in the case of vae ? 
        x6 = torch.flatten(x4, 1)
        if hasattr(self, 'dropout'):
            x6  = self.dropout(x6)
        #elif self.out_block == "contrastive":
        #    x6 = self.critic(x6)
        #    return x6
        #else:
        #    x6 = self.fc(x6).squeeze(dim=1)
        if self.linear_in_backbone:
            x6 = self.linear(x6)
        return x6


def _resnet(arch, block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(**kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2],  **kwargs)


## NB: initial_stride = 2 if initial_kernel_size==7 else 1
## stride 1 au début ? Modifier le kernel size en conséquence ? Voir si ça change l'architecture
# le soucis n'est pas le kernel size, il suffit de rajouter une couche au début sans stride.
# pas besoin de skip connections au début anyway.
# la taille du champ récepteur à la fin est-elle si importante ?

# in_channels=1
#num_classes = 1000 ? 256 ?
#doesn't matter, use 'SimCLR' config instead. Replace name with 'contrastive'.
#add linear towards backbone_output_size ?
#how is the projection head added in his framework ?