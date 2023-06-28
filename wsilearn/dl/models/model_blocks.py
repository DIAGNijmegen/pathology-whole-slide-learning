import torch
from torch import nn
from torch.nn import functional as F

from wsilearn.dl.models.model_utils import create_bn_act_drop_conv, create_conv_bn_act, create_conv, create_activation

#https://d2l.ai/chapter_convolutional-modern/resnet.html
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='same', act='relu', norm='bn',
                 dropout=0, **kwargs):
        super().__init__()
        self.conv1 = create_conv_bn_act(in_channels, out_channels, norm=norm, act=act, dropout=dropout,
                                        kernel_size=kernel_size, padding=padding, sequential=True, stride=stride,  **kwargs)
        self.conv2 = create_conv_bn_act(out_channels, out_channels, norm=norm, act=act,
                                        kernel_size=kernel_size, padding=padding, sequential=True, **kwargs)
        self.conv3 = None
        if in_channels != out_channels or stride!=1:
            self.conv3 = create_conv_bn_act(in_channels, out_channels, kernel_size=1, stride=stride,
                                            norm=norm, act=None, sequential=True, **kwargs)
        self.activation = create_activation(act)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.conv3 is not None:
            x = self.conv3(x)
        y = x+y
        y = self.activation(y)
        return y

class ResLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        out = self.layer(x)
        out = x+out
        return out

if __name__ == '__main__':
    rb = ResBlock(8, 8)
    print(rb)