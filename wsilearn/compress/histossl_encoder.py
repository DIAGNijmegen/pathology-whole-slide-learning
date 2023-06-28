import numpy as np
import os
from wsilearn.compress.encoders import PytorchEncoderWrapper

from pathlib import Path
histossl_enc_pathes = [
                    "~/.cache/torch/hub/checkpoints/histossl_tenpercent_resnet18_state_dict.ckpt",
                    "~/.torch/models/histossl_tenpercent_resnet18_state_dict.ckpt"
                    ]

import torchvision
import torch

def _get_histossl_encoder_path():
    for path in histossl_enc_pathes:
        path = os.path.expanduser(str(path))
        if Path(path).exists():
            return path
        else:
            print('checked %s' % str(path))
    return None

def _load_model_weights(model, weights):

    model_dict = model.state_dict()
    weights = {k: v for k, v in weights.items() if k in model_dict}
    if weights == {}:
        print('No weight could be loaded..')
    model_dict.update(weights)
    model.load_state_dict(model_dict)

    return model

class HistosslEncoderWrapper(PytorchEncoderWrapper):
    def __init__(self, model_path=None, preactivation=True, **kwargs):
        if model_path is None:
            model_path = _get_histossl_encoder_path()
        if model_path is None: raise ValueError('no histossl encoder!')
        self.preactivation = preactivation
        return super().__init__(normalizer='histossl', model_path=model_path, model_pred_fct=None, **kwargs)

    def _create_model(self):

        model = torchvision.models.__dict__['resnet18'](pretrained=False)
        model.load_state_dict(torch.load(self.model_path))
        model.fc = torch.nn.Sequential()
        # state = torch.load(self.model_path, map_location='cuda:0')
        return model

def _resave_model_dict():
    model_path = _get_histossl_encoder_path()
    model = torchvision.models.__dict__['resnet18'](pretrained=False)
    state = torch.load(model_path, map_location='cuda:0')
    state_dict = state['state_dict']
    for key in list(state_dict.keys()):
        state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)

    model = _load_model_weights(model, state_dict)


    torch.save(model.state_dict(), './out/histossl_state_dict.ckpt')

def _test_enc():
    enc = HistosslEncoderWrapper()
    # model = model.cuda()

    # patch_size = 224
    patch_size = 256
    # images = torch.rand((10, 3, patch_size, patch_size), device='cuda')

    images = np.random.rand(5, 256, 256, 3).astype(np.float32)
    out = enc(images)

    print(out.shape)

if __name__ == '__main__':
    # _resave_model_dict()
    _test_enc()