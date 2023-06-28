import torch
from torch import nn
import numpy as np
import torch.utils.checkpoint as cp
import torch.nn.functional as F

from wsilearn.utils.cool_utils import is_iterable
from wsilearn.dl.torch_utils import print_model_summary
from wsilearn.dl.models.model_utils import *

import warnings
# warnings.filterwarnings("error", category=UserWarning)

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, compression_factor=1.0, downsampling='avgpool', act='relu', norm=True):
        super().__init__()
        self.out_channels = int(np.round(in_channels*compression_factor))
        self.conv1 = create_bn_act_drop_conv(in_channels, self.out_channels, kernel_size=1, act=act, norm=norm)
        self.pool = create_pool(downsampling)

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool(out)
        return out

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=3, bottleneck=False, bottleneck_factor=4, dropout_type=None, dropout=0,
                 norm=True, padding='same', act='relu', sep_conv=False, mem_efficient=False):
        self.dropout = None
        super().__init__()
        self.bottle_conv = None
        next_channels = in_channels
        if bottleneck and int(bottleneck_factor*growth_rate) < next_channels:
            next_channels = int(bottleneck_factor*growth_rate)
            self.bottle_conv = create_bn_act_drop_conv(in_channels, next_channels, kernel_size=1, act=act,
                                                       norm=norm)
        self.conv = create_bn_act_drop_conv(next_channels, growth_rate,
                                            kernel_size=kernel_size, padding=padding,
                                            sep_conv=sep_conv, act=act,
                                            norm=norm, dropout_type=dropout_type, dropout=dropout)
        self.padding = padding
        self.mem_efficient = mem_efficient

    def forward(self, x):
        out = x
        if self.bottle_conv:
            if self.mem_efficient and self.training:
                with warnings.catch_warnings(record=True) as w:
                    out = cp.checkpoint(self.bottle_conv, out)
                    if len(w)>0:
                        print('checkpoint problem:', w[-1].category, w[-1].message)
                    else:
                        pass
            else:
                out = self.bottle_conv(out)
        out = self.conv(out)
        # if self.dropout is not None:
        #     out = self.dropout(out)
        if not self.padding:
            x = center_crop(x, out.shape[-2:])
        out = torch.cat((x, out), 1)
        return out

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        for l in range(n_layers):
            self.layers.append(DenseLayer(in_channels+growth_rate*l, growth_rate=growth_rate, **kwargs))

    def forward(self, x):
        for l,layer in enumerate(self.layers):
            x = layer(x)
        return x

class DenseNet(nn.Module):
    def __init__(self, block_sizes, out_dim, in_dim=3, depth=None, bottleneck=True, bottleneck_factor=4,
                 growth_rate=12, compression_factor=1.0, pool='avgpool', padding=True,
                 dropout_layers=0, dropout=0, dropout_input=False, dropout_type=None,
                 first_layer_str=None, first_layer_str_short='', name='dense', first_features=None, act='relu',
                 keep_last_representation=False, sep_conv=False, mem_efficient=False,
                 final_pool='avg_pool', final_bottleneck=0, final_dropout=0,
                 norm=True,  #attention=None, attention_input=None,
                 kernel_size=3, **kwargs):
        #FIXME: rename attention to final_attention
        super().__init__()
        print('Densenet ignoring following kwargs:', kwargs)
        print('blocks sizes %s, growth rate %d, mem_efficient=%s' % (str(block_sizes), growth_rate, str(mem_efficient)))
        if not is_iterable(block_sizes):
            block_sizes = [block_sizes]
        if depth is not None and depth>0:
            if len(block_sizes)==1 and depth>=1:
                block_sizes = block_sizes*depth
            else:
                raise ValueError('invalid block_sizes and depth:',block_sizes, depth)

        # self.att_inp = None
        # if attention_input is not None:
        #     self.att_inp = create_attention(attention_input, in_channels, bn=bn)

        #conv3s1p1
        if first_layer_str is None or str(first_layer_str).lower()=='none' or len(str(first_layer_str))==0:
            first_layer_str = 'conv7s2p3_bn_relu_maxpool3s2p1'
            first_layer_str_short = ''
        if str(first_layer_str_short) in ['None','x']:
            first_layer_str_short = first_layer_str

        self.keep_last_representation = keep_last_representation
        if first_features is None:
            next_channels = 2*growth_rate
        else:
            next_channels = first_features

        self.dropout_input = None
        if dropout_input:
            self.dropout_input = create_dropout(dropout, dropout_type)

        self.conv1 = create_layer_from_string(first_layer_str, in_channels=in_dim, out_channels=next_channels)
        if self.conv1 is None:
            next_channels = in_dim

        dropout_blocks = []
        for i in range(dropout_layers):
            dropout_blocks.append(len(block_sizes)-i-1)
        self.layers = nn.Sequential()
        for i,block_size in enumerate(block_sizes):
            mem_eff_i = mem_efficient
            if i==0 and first_layer_str is None:
                mem_eff_i = False
                if mem_efficient: print('making block %d not mem_efficient' % i)
            block = DenseBlock(in_channels=next_channels, growth_rate=growth_rate, n_layers=block_size,
                               dropout_type=dropout_type if i in dropout_blocks else None, dropout=dropout,
                               bottleneck=bottleneck, bottleneck_factor=bottleneck_factor, kernel_size=kernel_size,
                               padding=padding, sep_conv=sep_conv, mem_efficient=mem_eff_i, norm=norm)
            self.layers.add_module('block%d' % (i + 1), block)
            next_channels = next_channels + block_size * growth_rate
            if i < len(block_sizes)-1:
                transition_layer = TransitionLayer(next_channels, compression_factor=compression_factor,
                                                   downsampling=pool, norm=norm, act=act)
                next_channels = transition_layer.out_channels
                self.layers.add_module('transition%d' % (i + 1), transition_layer)

        final_ops = [create_bn(norm, next_channels), create_activation(act)]
        # if attention is not None:
        #     final_ops.append(create_attention(attention, next_channels, bn=bn))
        if final_pool:
            final_ops.append(create_pool(final_pool))
        if final_bottleneck and next_channels>final_bottleneck:
            final_ops.extend(create_conv_bn_act(next_channels, final_bottleneck, kernel_size=1,
                                                activation=act, as_sequential=False, bn=norm))
            next_channels = final_bottleneck
        if final_dropout:
            final_ops.append(create_dropout(dropout_type='dropout', dropoute=final_dropout))

        self.last_op = nn.Sequential(*final_ops)

        self.fc = nn.Conv2d(in_channels=next_channels, out_channels=out_dim,
                            bias=True, kernel_size=1)

        #create short name
        parts = [name, first_layer_str_short]
        if first_features != 2*growth_rate:
            parts.append('f%d' % first_features)
        if kernel_size!=3:
            parts.append('ks%d' % kernel_size)
        parts.append('b'+''.join(str(bl) for bl in block_sizes))
        if growth_rate is not None and growth_rate > 0:
            parts.append('g%d' % growth_rate)
        if sep_conv: parts.append('sc')
        if dropout_type is not None and dropout:
            drop_str = 'dr'
            if dropout_layers:
                drop_str+=f'l{dropout_layers}'
            if dropout_input is not None:
                drop_str+='i'
            parts.append('%s%d' % (drop_str, int(dropout*100)))
        if act != 'relu':
            parts.append(act)
        parts.append(final_pool)
        if final_bottleneck is not None and final_bottleneck!=0:
            parts.append('fb%d' % final_bottleneck)
        # if attention is not None:
        #     stri+='_%s-att' % attention
        if final_dropout:
            parts.append('fdr%.1d' % final_dropout)
        self.short_name='_'.join(parts)
        self.short_name = self.short_name.replace('__','_')

    def forward(self, x):
        out = x
        if self.dropout_input is not None:
            out = self.dropout_input(out)
        # if self.att_inp is not None:
        #     out = self.att_inp(out)
        if self.conv1 is not None:
            out = self.conv1(out)
        out = self.layers(out)
        out = self.last_op(out)
        if self.keep_last_representation:
            self.last_representation = out
        out = self.fc(out)
        out = out.squeeze(-1).squeeze(-1)
        return out



class DenseNet121(DenseNet):
    def __init__(self, **kwargs):
        super().__init__(block_sizes=(6, 12, 24, 16), growth_rate=32, first_features=64, bottleneck=True,
                         compression_factor=0.5, name='dense121', **kwargs)

class DenseNetMini(DenseNet):
    def __init__(self, **kwargs):
        super().__init__(block_sizes=[2, 2], growth_rate=12, name='dense_mini', **kwargs)

class DenseNet28BC(DenseNet):
    def __init__(self, **kwargs):
        super().__init__(block_sizes=[4, 4, 4], growth_rate=32, compression_factor=0.5, bottleneck_factor=2,
                         name='dense28bc', **kwargs)

def pytorch_densenet121(pretrained=False, **kwargs):
    from torchvision.models.densenet import densenet121
    net = densenet121(pretrained=pretrained, **kwargs)
    return net

if __name__ == '__main__':
    # in_channels = 3
    # in_shape = [in_channels, 224, 224]
    # mynet = DenseNet121(mem_efficient=True, out_dim=2, in_channels=in_channels)
    # mynet = DenseNetMini(out_dim=2)
    # print('my densenet:')
    # print(mynet)
    # inp = torch.zeros([1, in_channels, 224, 224])
    # out = mynet(inp)
    # print(inp.shape, out.shape)

    # mynet = DenseNet28BC()
    # print_model_summary(mynet, in_shape)
    # tnet = pytorch_densenet121(num_classes=10)
    # print('torch densenet:')
    # print_model_summary(tnet, in_shape)

    net = DenseNet(first_layer_str='conv1s1p0_bn_relu', first_layer_str_short='',
                   padding=0, in_dim=1024, out_dim=2, first_features=512,
                   block_sizes=[1, 1, 1], compression_factor=0.5, bottleneck_factor=2,
                   final_pool='gmax', sep_conv=True
                   )
    # print(net)
    print_model_summary(net, [1, 1024, 50, 50])
    print(net.short_name)