from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from wsilearn.compress.mtdp_encoder import MtdpEncoderWrapper
from wsilearn.utils.cool_utils import can_open_file
from wsilearn.dl.torch_utils import print_model_summary
from wsilearn.compress.encoders import  TimmEncoder, IncRes2Encoder, Res50Encoder, DensenetEncoder


def create_encoder(encoder, encoder_path=None, **enc_kwargs):
    if 'histossl'== encoder:
        from wsilearn.compress.histossl_encoder import HistosslEncoderWrapper
        enc = HistosslEncoderWrapper(model_path=encoder_path)
        print_model_summary(enc._create_model(), torch.zeros((2,3,256,256)))
    elif "mtdp" == encoder or 'mtdp_resnet' in encoder or 'mtdp_res50' in encoder:
        enc = MtdpEncoderWrapper(arch="resnet50", model_path=encoder_path)
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
    elif 'ctran' in encoder:
        from wsilearn.compress.ctran.ctran_encoder import CtranEncoder
        enc = CtranEncoder(model=encoder_path, **enc_kwargs)
    else:
        raise ValueError('unknown encoder %s' % encoder)

    return enc



if __name__ == '__main__':
    # enc = create_encoder('vicreg_res50')
    # enc = create_encoder('hipt')
    # enc = create_encoder('res50')
    enc = create_encoder('mtdp_res50')
    # enc = create_encoder('lunitbt')
    inp = np.zeros((6,256,256,3))
    out = enc(inp)
    print(inp.shape, out.shape, out.dtype)