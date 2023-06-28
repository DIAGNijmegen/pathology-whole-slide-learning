from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from wsilearn.utils.cool_utils import can_open_file
from wsilearn.dl.torch_utils import print_model_summary
from wsilearn.utils.gpu_utils import count_gpus
from wsilearn.utils.ray_utils import init_ray_on_docker, CallRayWrapper
from wsilearn.compress.encoders import LighntningUnsupervisedEncoderWrapper, \
    MtdpEncoderWrapper, TimmEncoder, IncRes2Encoder, Res50Encoder, DensenetEncoder


def create_encoder(encoder, encoder_path=None, layer_name=None, multiproc=False,
                   n_cpus=None, n_gpus=None, model_pred=None, **enc_kwargs):
    if 'plbolts' in encoder or 'pl-bolts' in encoder:
        enc = LighntningUnsupervisedEncoderWrapper(name=encoder)
    elif 'histossl'== encoder:
        from wsilearn.compress.histossl_encoder import HistosslEncoderWrapper
        enc = HistosslEncoderWrapper(model_path=encoder_path)
        print_model_summary(enc._create_model(), torch.zeros((2,3,256,256)))
    elif "mtdp" == encoder or 'mtdp_resnet' in encoder or 'mtdp_res50' in encoder:
        enc = MtdpEncoderWrapper(arch="resnet50")
    elif 'mtdp_densenet'in encoder:
        enc = MtdpEncoderWrapper(arch="densenet121")
    elif 'incres2' == encoder:
        enc = IncRes2Encoder()
    elif 'res50' == encoder:
        enc = Res50Encoder()
    elif 'densenet' == encoder:
        enc = DensenetEncoder()
    elif 'densenet_last' == encoder:
        enc = DensenetEncoder(last=True)
    elif 'res50last' == encoder:
        enc = Res50Encoder(last=True)
    else:
        raise ValueError('unknown encoder %s' % encoder)

    print(f'multiproc {multiproc}, n_cpus: {n_cpus}, n_gpus: {n_gpus}')
    if multiproc:
        if n_gpus is None:
            n_gpus = count_gpus()
        # encoder = RayCallWrapper.remote(encoder)
        print(f'calling init_ray_on_docker with {n_cpus} cpus and {n_gpus} gpus')
        cpus, gpus = init_ray_on_docker(n_cpus=n_cpus, n_gpus=n_gpus)
        # gpus = torch.cuda.device_count()
        print('creating CallRayWrapper with %d gpus' % n_gpus)
        enc = CallRayWrapper(enc, n_gpus=n_gpus)
    return enc



if __name__ == '__main__':
    # enc = create_encoder('vicreg_res50')
    # enc = create_encoder('hipt')
    enc = create_encoder('res50')
    # enc = create_encoder('lunitbt')
    inp = np.zeros((6,256,256,3))
    out = enc(inp)
    print(inp.shape, out.shape, out.dtype)