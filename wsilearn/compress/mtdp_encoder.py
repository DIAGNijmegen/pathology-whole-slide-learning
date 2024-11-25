
# Adapted from https://github.com/waliens/multitask-dipath, since the original code is not compatible
# with current pytorch version


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import torch
from abc import abstractmethod
from torch import nn


from torchvision.models.resnet import Bottleneck, ResNet

from wsilearn.compress.encoders import PytorchEncoderWrapper
from wsilearn.utils.cool_utils import showim, showims
from wsilearn.utils.df_utils import df_read, print_df



class FeaturesInterface(object):
    @abstractmethod
    def n_features(self):
        pass


class NoHeadResNet(ResNet, FeaturesInterface):
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)

    def n_features(self):
        return [b for b in list(self.layer4[-1].children()) if hasattr(b, 'num_features')][-1].num_features



class PooledFeatureExtractor(nn.Module, FeaturesInterface):
    """This module applies a global average pooling on features produced by a module.
    """

    def __init__(self, features):
        """
        Parameters
        ----------
        features: nn.Module
            A network producing a set of feature maps. `features` should have a `n_features()` method
            returning how many features maps it produces.
        """
        super().__init__()
        self.features = features
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return self.pool(self.features(x))

    def n_features(self):
        return self.features.n_features()

""" MTDP ResNet-50 encoder """
class MtdpEncoderWrapper(PytorchEncoderWrapper):
    def __init__(self, model_path=None, model_pred_fct='id', arch="resnet50", **kwargs):
        self.arch = arch
        if model_path is None:
            model_path = '~/.torch/models/resnet50-mh-best-191205-141200.pth'
            model_path = os.path.expanduser(model_path)
            if not Path(model_path).exists():
                raise ValueError('model not found at %s. MTDP model available from https://github.com/waliens/multitask-dipath' % model_path)
        return super().__init__(normalizer='imagenet', model=model_path, model_pred_fct=model_pred_fct, **kwargs)

    def _create_model(self):
        # from mtdp import build_model #requires the mtdp code
        # from mtdp.models._util import clean_state_dict
        state_dict = torch.load(self.model, map_location='cpu')
        state_dict = clean_state_dict(state_dict, prefix="features.", filter=lambda k: not k.startswith("heads."))
        model = NoHeadResNet(Bottleneck, [3, 4, 6, 3])
        # model = build_model(arch=self.arch, pretrained="mtdp", pool=self.pool)
        model.load_state_dict(state_dict)
        model = PooledFeatureExtractor(model)
        return model

def clean_state_dict(state_dict, prefix, filter=None):
    if filter is None:
        filter = lambda *args: True
    return {_remove_prefix(k, prefix): v for k, v in state_dict.items() if filter(k)}

def _remove_prefix(s, prefix):
    if s.startswith(prefix):
        s = s[len(prefix):]
    return s


if __name__ == '__main__':
    model = MtdpEncoderWrapper()
