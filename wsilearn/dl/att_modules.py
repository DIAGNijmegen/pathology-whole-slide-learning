from functools import partial

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16, ret_att=False):
        super().__init__()
        self.ret_att = ret_att
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        channels_reduced = max(4, channel // reduction) #saveguarded to have at least four channel
        self.fc = nn.Sequential(
            nn.Linear(channel, channels_reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels_reduced, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        size = list(x.size())
        b = size[0]; c = size[1]
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y)
        if self.ret_att:
            return y
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)

class ChannelAttentionBAM1d(nn.Module):
    """ Adapted from CBAM (ChannelGate) """
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.gate_channels = in_channels
        n_hidden = max(1, in_channels // reduction_ratio)
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(in_channels, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, in_channels)
        )

    def forward(self, x):
        A = self.mlp(x)
        scale = torch.sigmoid(A)
        return x * scale


class ChannelConstant1d(nn.Module):
    """ Adapted from CBAM (ChannelGate) """
    def __init__(self, in_channels):
        super().__init__()
        self.att_vec = self.weight = Parameter(torch.ones(in_channels))

    def forward(self, x):
        scale = torch.sigmoid(self.att_vec)
        return x * scale

#ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks
#https://github.com/BangguWu/ECANet/blob/master/models/eca_module.py
# Changes: rename eca_layer to ChannelAttentionLayer)
# code in the repo different from paper... this is the original
class ChannelAttentionLayer(nn.Module):
    """Constructs a ECA module.
    Args:
        #channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

#from cbam
class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, max(8,gate_channels // reduction_ratio)),
            nn.ReLU(),
            nn.Linear(max(8,gate_channels // reduction_ratio), gate_channels)
        )
        self.pool_types = pool_types
        self.short_name = 'chg'

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class ChannelGateMP(ChannelGate):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super().__init__(gate_channels, reduction_ratio, pool_types=['max'])
        self.short_name='chgmp'

def create_attention(name, in_dim=None, **kwargs):
    # if bn is None or (is_string(bn) and bn in ['gn', 'bn']):
    #     bn = True
    # elif is_string(bn):#for all other also use that bn in attention layers
    #     bn = partial(create_bn, bn)
    #     if name=='self': raise ValueError('implement using bn-fct in self attention')

    if name is None:
        return None

    if name=='ch' or name =='channel':
        return ChannelAttentionLayer(**kwargs)
    elif name.lower() in ['chg','channelgate']:
        return ChannelGate(in_dim, **kwargs)
    elif name.lower() in ['chgmp','channelgatemp']:
        return ChannelGateMP(in_dim, **kwargs)
    # elif name=='self':
    #     if in_channels < 8: raise ValueError('too few channels %d for self attention, must be at least 8' % in_channels)
    #     return SelfAttention(in_channels, **kwargs)
    # elif name=='cbam':
    #     return CBAM(in_channels, bn=bn)
    # elif name=='bam':
    #     return BAM(in_channels, bn=bn)
    # elif name=='bamnr':
    #     return BAM(in_channels, residual=False, bn=bn)
    # elif name == 'context':
    #     return ContextAttentionLayer(in_channels, **kwargs)
    # elif name == 'eca':
    #     return ECA(in_channels, **kwargs)
    elif name == 'se':
        return SELayer(in_dim, **kwargs)
    else: raise ValueError('unknown attention name %s' % name)

if __name__ == '__main__':
    torch.manual_seed(1)

    # cha = ChannelAttentionBAM1d(32)
    # inp = torch.zeros((2, 32))
    # out = cha(inp)

    cha = ChannelConstant1d(32)
    inp = torch.rand((2, 32))
    out = cha(inp)
