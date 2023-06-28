import logging, re
# from logging import info

import functools
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn.modules.padding import ReflectionPad2d, ZeroPad2d
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR

from wsilearn.utils.cool_utils import is_callable, is_string, is_list_or_tuple
from collections import OrderedDict

from wsilearn.dl.att_modules import create_attention
from wsilearn.dl.pool_utils import create_pool
from wsilearn.dl.torch_utils import to_numpy
from wsilearn.dataconf import TrainType

def weights_init_module(m, mode='fan_in', act='linear', lrelu_a=0):
    """ Init layer parameters """
    if act in ['lrelu','leaky_relu']:
        nonlinearity = 'leaky_relu'
    elif act in ['tanh', 'relu']:
        nonlinearity = act
    else:
        #including selu to achieve self-normalization (but with 'selu' more stable gradients)
        nonlinearity = 'linear'

    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode=mode, nonlinearity=nonlinearity, a=lrelu_a)
        if not (m.bias is None or m.bias is False):
            nn.init.constant_(m.bias, 0)
    # elif isinstance(m, nn.Linear):
    #     nn.init.kaiming_uniform_(m.weight)
    #     if not (m.bias is None or m.bias is False):
    #         nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def weights_init(net, act='leaky_relu', mode='fan_in', **kwargs):
    # print('init weights with nonlinearity %s' % nonlinearity)
    if is_list_or_tuple(net):
        iterator = net
    else:
        iterator = net.modules()
    for i,m in enumerate(iterator):
        weights_init_module(m, mode=mode, act=act, **kwargs)

class WeightsInit(object):
    def __init__(self, act='linear', **kwargs):
        self.act = act
        self.kwargs = kwargs
    def __call__(self, net):
        weights_init(net, act=self.act, **self.kwargs)

def cmp_out_shape_dyn(in_shape, module):
    print('computing output shape for input shape %s' % str(in_shape))
    input_shape = [2]+list(in_shape)
    inp = torch.zeros(input_shape)
    # if module.is_cuda():
    #     inp = inp.cuda()
    with torch.no_grad():
        out = module(inp)
    return list(out.shape)[1:]

def center_crop(layer, target_size):
    _, _, layer_height, layer_width = layer.size()
    if layer_height == target_size[0] and layer_width == target_size[1]:
        return layer
    diff_y = (layer_height - target_size[0]) // 2
    diff_x = (layer_width - target_size[1]) // 2
    return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]


def create_sep_conv(in_channels, out_channels, kernel_size, padding=0, **kwargs):
    dkwargs = deepcopy(kwargs)
    dkwargs['groups'] = in_channels
    return nn.Sequential(OrderedDict([
        ('depthwise', nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, **dkwargs)),
        ('pointwise', nn.Conv2d(in_channels, out_channels, kernel_size=1, **kwargs))]))


def create_activation(act_name, lrelu_a=0.2, **kwargs):
    if act_name == 'relu':
        return nn.ReLU(**kwargs)
    elif act_name == 'leaky_relu' or act_name=='lrelu':
        return nn.LeakyReLU(lrelu_a, **kwargs)
    elif act_name=='prelu':
        return nn.PReLU(**kwargs)
    elif act_name=='selu':
        return nn.SELU(**kwargs)
    elif act_name=='gelu':
        return nn.GELU()
    elif act_name in ['hardswish', 'hswish']:
        return nn.Hardswish(**kwargs)
    elif act_name=='mish':
        return nn.Mish(**kwargs)
    elif act_name in ['silu','swish']:
        return nn.SiLU(**kwargs)
    elif act_name in ['sig', 'sigm', 'sigmoid']:
        return nn.Sigmoid()
    elif act_name in ['tanh']:
        return nn.Tanh()
    else:
        raise ValueError('Unknown activation %s' % act_name)

def create_conv(in_channels, out_channels, kernel_size, sep_conv=False, stride=1, padding=0,
                in_size=None, init=False, **kwargs):
    if padding is None or padding=='valid' or kernel_size==1:
        padding = 0
    if padding=='same':
        if stride!=1 and in_size is not None:
            padding = (in_size - kernel_size)//stride + 1
        else:
            padding = (kernel_size-1)//2
    else:
        padding = int(padding)

    if sep_conv and kernel_size>1:
        conv = create_sep_conv(in_channels, out_channels, kernel_size=kernel_size, padding=padding,
                               stride=stride, **kwargs)
    else:
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, **kwargs)
    if init:
        weights_init_module(conv, act=None)
    return conv

def create_upsampling(name, in_ch, out_ch, upsample_kernel=3, upsample_padding=1):
    if name in ['transposedconv']:
        up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
    elif name in ['upsample', 'upsampling2d', 'upsampling']:
        up = nn.Sequential(nn.Upsample(mode='nearest', scale_factor=2),
                           create_conv(in_ch, out_ch, kernel_size=upsample_kernel, padding=upsample_padding))
    else:
        raise ValueError('unknown upsampling method %s' % name)
    return up

def stable_softmax(inp, dim=None, temperature=None):
    if temperature not in [None, 1]:
        if temperature == 'sqrt':
            inp = inp/(torch.sum(torch.isfinite(inp), dim=dim, keepdim=True).sqrt())
        else:
            inp = inp/temperature
    return torch.nn.LogSoftmax(dim=dim)(inp).exp()

""" 1x1 convolution """
create_conv1 = functools.partial(create_conv, kernel_size=1)

#manual batchnorm impl:
#https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py
def create_bn(bn, channels:int, dims=2, **kwargs):
    """ bn: BatchNorm, in: InstanceNorm, gn: GroupNorm with 32 groups, gn<#groups>: GroupNorm with #groups"""
    if is_callable(bn):
        return bn(channels)

    if (not is_string(bn) and bn) or bn=='bn' or bn==True:
        if dims==1:
            bn = nn.BatchNorm1d(channels)
        elif dims==2:
            bn = nn.BatchNorm2d(channels)
        else:
            bn = nn.BatchNorm3d(channels)
    elif bn == 'in':
        if dims==1:
            bn = nn.InstanceNorm1d(channels)
        elif dims==2:
            bn = nn.InstanceNorm2d(channels)
        else:
            bn = nn.InstanceNorm3d(channels)
    elif bn == 'ln':
        bn = nn.GroupNorm(1, channels)
    elif bn.startswith('gn'):
        if len(bn)==2:
            groups = 32
        else:
            groups = int(bn[2:])
        if groups > channels//2:
            groups = channels//2
        if groups<=0:
            groups = 1
        bn = nn.GroupNorm(groups, channels)
    else:
        raise ValueError('unknown bn %s' % str(bn))
    return bn


def create_lr_scheduler(name, optimizer, **kwargs):
    if name=='plateau':
        scheduler = ReduceLROnPlateau(optimizer, **kwargs)
    elif name=='cyclic':
        scheduler = CyclicLR(optimizer, **kwargs)
    else:
        raise ValueError('unknown lr scheduler %s' % name)
    return scheduler

def create_optimizer(name, model_parameters, lr, weight_decay=0):
    if name == 'adam':
        optimizer = torch.optim.Adam(model_parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adamw':
        optimizer = torch.optim.AdamW(model_parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'radam':
        optimizer = torch.optim.RAdam(model_parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'nadam':
        optimizer = torch.optim.NAdam(model_parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'sgd':
        optimizer = torch.optim.SGD(model_parameters, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    elif callable(name):
        optimizer = name(model_parameters)
    else:
        raise ValueError('unknown optimizer %s' % name)

    return optimizer

# def create_optimizer(optimizer, model, lr, weight_decay=0, final_lr=0.1):
#     if optimizer == 'adam':
#         optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
#     elif optimizer == 'adamw':
#         optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
#     elif optimizer == 'sgd':
#         optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay,
#                                     nesterov=True)
#     elif optimizer == 'adabelief':
#         from adabelief_pytorch import AdaBelief
#         optimizer = AdaBelief(model.parameters(), lr=lr, eps=1e-16, betas=(0.9, 0.999), weight_decouple=True,
#                               rectify=False)
#     elif optimizer == 'adabound':
#         from adabound import AdaBound
#         optimizer = AdaBound(model.parameters(), lr=lr, final_lr=final_lr, weight_decay=weight_decay)
#     elif callable(optimizer):
#         optimizer = optimizer(model)
#     else:
#         raise ValueError('unknown optimizer %s' % optimizer)
#
#     return optimizer

def create_dropout(dropout, dropout_type='dropout', block_size=4, **kwargs):
    if dropout_type is None or not dropout_type:
        return None

    if dropout <= 0 or dropout>=1: raise ValueError(f'dropout rate {dropout}!')

    if dropout_type=='dropout':
        drop = nn.Dropout(dropout, **kwargs)
        drop.short_name='dr%d' % (dropout*100)
    elif dropout_type in ['spatial', 'dropout2d', 'drop2d']:
        drop = nn.Dropout2d(dropout, **kwargs)
        drop.short_name='dr2d%d' % (dropout*100)
    elif dropout_type in ['dropout_block2d', 'drop_block2d', 'block2d']:
        from dropblock import DropBlock2D
        drop = DropBlock2D(dropout, block_size=block_size)
        drop.short_name='dr2d%d' % (dropout*100)
    elif 'block' in dropout_type:
        drop = torchvision.ops.drop_block2d()
    else:
        raise ValueError('unknonw dropout %s' % dropout_type)
    return drop

def create_conv_bn_act(in_channels, out_channels, kernel_size, act='relu', norm=True, att=None,
                       dropout=0, dropout_type='dropout', sequential=False, lrelu_a=0.2, **kwargs):

    modules = []
    if att is not None:
        att = create_attention(att, in_channels)
        modules.append(att)
    conv = create_conv(in_channels, out_channels, kernel_size=kernel_size, bias=not norm, **kwargs)
    modules.append(conv)
    if norm:
        modules.append(create_bn(norm, out_channels))
    if act:
        modules.append(create_activation(act, lrelu_a=lrelu_a))
    if dropout:
        modules.append(create_dropout(dropout, dropout_type=dropout_type))
    if sequential:
        modules = nn.Sequential(*modules)

    weights_init(modules, act=act, lrelu_a=lrelu_a)
    return modules

def create_conv_act_bn(in_channels, out_channels, kernel_size, act='relu', norm=True, att=None,
                       dropout=0, dropout_type='dropout', sequential=False, lrelu_a=0.2, **kwargs):

    modules = []
    if att is not None:
        att = create_attention(att, in_channels)
        modules.append(att)
    conv = create_conv(in_channels, out_channels, kernel_size=kernel_size, **kwargs)
    modules.append(conv)
    if act:
        modules.append(create_activation(act, channels=out_channels, lrelu_a=lrelu_a))
    if norm:
        modules.append(create_bn(norm, out_channels))
    if dropout:
        modules.append(create_dropout(dropout, dropout_type=dropout_type))
    if sequential:
        modules = nn.Sequential(*modules)

    weights_init(modules, act=act, lrelu_a=lrelu_a)
    return modules

def create_bn_act_drop_conv(in_channels, out_channels, kernel_size=3, act='relu', norm=True,
                            sequential=True, dropout=0, dropout_type='dropout', lrelu_a=0.2, **kwargs):
    modules = []
    if norm:
        modules.append(create_bn(norm, in_channels))
    if act is not None:
        modules.append(create_activation(act))
    if dropout_type is not None and dropout_type and dropout:
        modules.append(create_dropout(dropout_type, dropout))
    conv = create_conv(in_channels, out_channels, kernel_size=kernel_size, bias=not norm, **kwargs)
    modules.append(conv)
    if sequential:
        modules = nn.Sequential(*modules)
    weights_init(modules, act=act, lrelu_a=lrelu_a)
    return modules

def create_linear_act(in_channels, out_channels, act=None, norm=True,
                            sequential=False, dropout=0, dropout_type='dropout'):
    layers = []
    layers.append(nn.Linear(in_channels, out_channels, bias=not norm))
    layers.append(create_activation(act))
    if norm:
        layers.append(create_bn(norm, out_channels, dims=1))
    if dropout:
        layers.append(create_dropout(dropout, dropout_type=dropout_type))

    if sequential:
        layers = nn.Sequential(*layers)
    weights_init(layers, act=act)
    return layers

def create_linear(in_channels, out_channels, bias=True, **kwargs):
    layer = nn.Linear(in_channels, out_channels, bias=bias, **kwargs)
    weights_init_module(layer, act=None)
    return layer

def create_layer_from_string(stri, in_channels, out_channels):
    """ example: bn_conv3s1p0_relu_maxpool2"""
    if stri is None:
        return None
    layers = []
    bn = False
    if '_bn_' in stri or '_bn' in stri or 'bn_' in stri:
        bn = True
    parts = stri.split('_')
    current_in_channels = in_channels
    for pa,part in enumerate(parts):
        if part in ['bn','gn','ln']:
            layers.append(create_bn(part, current_in_channels))
        elif part.startswith('conv'):
            #e.g. conv3s2 - kernel size 3, stride 2
            p = re.compile('conv(\d+)s(\d+)p(\d+)')
            m = re.match(p, part)
            if not m: raise ValueError('unknown part of convolution layer format string %s' % part)
            kernel_size, stride, padding = m.groups()
            kernel_size = int(kernel_size); stride = int(stride); padding = int(padding)
            layers.append(nn.Conv2d(current_in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                    padding=padding, bias=not bn))
            current_in_channels = out_channels
        elif part.startswith('maxpool') or part.startswith('avgpool'):
            p = re.compile('(\w+)(\d+)s(\d+)p(\d+)')
            m = re.match(p, part)
            if not m: raise ValueError('unknown part of pooling layer format string %s' % part)
            pool_name, kernel_size, stride, padding = m.groups()
            kernel_size = int(kernel_size); stride = int(stride); padding = int(padding)
            layers.append(create_pool(pool_name, kernel_size=kernel_size, stride=stride, padding=padding))
        elif 'elu' in part:
            layers.append(create_activation(part))
        else:
            raise ValueError('uknown part of layer format string %s' % part)
    return nn.Sequential(*layers)

def clf_prediction(logits, targets=None, train_type=TrainType.clf):
    """ logits and targets (optional) tensors
    returns the predictions and targets (optional) as labels """
    if train_type == TrainType.regr:
        return None,  to_numpy(logits)

    multilabel = train_type == TrainType.multilabel
    if multilabel:
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).to(torch.float32)
    else:
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
    preds = to_numpy(preds)
    if targets is not None:
        targets = torch.argmax(targets, dim=1)
        return to_numpy(probs), preds, to_numpy(targets)
    return to_numpy(probs), preds

class ModuleConcat(nn.Module):
    def __init__(self, module_class, n, dim, **kwargs):
        super().__init__()
        self.dim = dim
        self.module_list = nn.ModuleList()
        for i in range(n):
            m = module_class(**kwargs)
            self.module_list.append(m)
            if i==0 and hasattr(m, 'short_name'):
                self.short_name = m.short_name

    def forward(self, x):
        outs = []
        for m in self.module_list:
            outs.append(m(x))
        out = torch.cat(outs, self.dim)
        return out

if __name__ == '__main__':
    inp = torch.zeros((1, 10, 32, 32))
    # layers = create_layer_from_string('conv3s2p0_bn_relu_maxpool3s2p1', in_channels=10, out_channels=10)
    # print(layers)

    # conv = create_conv_bn_act(10, 5, 2, sequential=True)
    # print(conv)
    # out = conv(inp)
    # print(out.shape)

    lin = create_linear(16, 32)