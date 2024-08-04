import numpy as np

from wsilearn.utils.cool_utils import *
from wsilearn.dl.np_normalizer import create_input_normalizer
from wsilearn.dl.models.model_internal_utils import ModelHook
from wsilearn.dl.np_transforms import NumpyToTensor
from wsilearn.dl.torch_utils import *

import torch.nn.functional as F
import torchvision.models

from wsilearn.utils.cool_utils import print_mp

print = print_mp


class PatchStatEncoder(object):
    """ encodes each dimension of the patch into a histogram or other statistical encoding (mean, std, etc.)"""

    def __call__(self, arrs):
        # hwc
        single_input = False
        if len(arrs.shape)==3: #single input -> batch
            arrs = arrs[None,:,:,:]
            single_input = True
        elif len(arrs.shape)<3:
            raise ValueError('expects arrays to have shape 3 or 4, not %s' % str(arrs.shape))

        encoded = []
        for arr in arrs:
            enc = []
            dims = arr.shape[-1]
            non_zero_ind = np.abs(arr).sum(-1) != 0
            for i in range(arr.shape[-1]):
                enc_i = self._encode(arr[:,:,i][non_zero_ind])
                enc.append(enc_i)
            enc = np.concatenate(enc)
            encoded.append(enc)

        if single_input:
            return encoded[0]
        else:
            return np.vstack(encoded)


class MeanStdEncoder(PatchStatEncoder):
    def __init__(self, n_bins=10, range=(0,1)):
        self.n_bins = n_bins
        self.range = range

    def _encode(self, arr):
        return np.array([arr.mean(), arr.std()])


class EncoderWrapper(object):
    def __init__(self, normalizer, model=None, hist_bins=None, stat_enc=None, pred_center_crop=None, hwc=True,
                 verbose=True, model_name=None, **stat_enc_kwargs):
        self.normalizer = create_input_normalizer(normalizer)
        self.model_name = model_name
        if is_string_or_path(model) and '~' in str(model):
            model = os.path.expanduser(str(model))
        self.model = model
        self.calls = 0
        self.channels_first = None
        self.verbose = verbose

        self.hist_bins = hist_bins
        self.network = None
        self.pred_center_crop = pred_center_crop

        self.stat_enc = None
        self.hwc = hwc
        if stat_enc is None:
            self.stat_enc = None
        elif stat_enc == 'mean_std':
            print('stat_enc is MeanStdEncoder')
            self.stat_enc = MeanStdEncoder(**stat_enc_kwargs)
        elif stat_enc == 'hist':
            print('stat_enc is HistogramEncoder')
            self.stat_enc = HistogramEncoder(**stat_enc_kwargs)
        else:
            print('stat_enc_kwargs',stat_enc_kwargs)
            raise ValueError('unknown value for stat_enc: %s' % str(stat_enc))

    def _encode(self, data):
        raise ValueError('implement _encoder')

    def __call__(self, data):
        """ assumes data is in hwc format """
        data = self.normalizer(data)

        if self.network is None:
            self.network = self._init_network()
        """ prints the model summary on first call """
        if not self.hwc:
            data = hwc_to_chw(data)

        if self.calls == 0:
            dummy_data = np.zeros_like(data)
            if len(data.shape)==4 and len(data)>2:
                #batch of 2 is enough
                dummy_data = dummy_data[-2:]
            if self.verbose:
                self.print_model_summary(dummy_data)
        self.calls += 1

        pred = self._encode(data)
        if self.pred_center_crop is not None:
            if len(pred.shape)==3:
                pred = center_crop_img(pred, self.pred_center_crop)
            elif len(pred.shape)==4:
                cropped = [center_crop_img(p, self.pred_center_crop) for p in pred]
                pred = np.array(cropped)
            else: raise ValueError('unknown pred shape for center cropping: %s' % str(pred.shape) )

        if self.stat_enc is not None:
            pred = self.stat_enc(pred)
        return pred
        # invert_op = getattr(encoder, "print_model_summary", None)
        # if callable(invert_op):
        #     encoder.print_model_summary([input_size, input_size, 3])

    def print_model_summary(self, input_shape=None):
        if self.network is None:
            self.network = self._init_network()
        return self._print_model_summary(input_shape)

    def _init_network(self):
        if not is_string_or_path(self.model):
            return self.model
        raise ValueError('implemement _init_network')

    def get_network(self):
        if self.network is None:
            self.network = self._init_network()
        return self.network

class PytorchEncoderWrapper(EncoderWrapper):
    def __init__(self, model=None, model_pred_fct=False, eval=True,
                 layer=None, layer_name=None, hwc=False, avgpool=False, **kwargs):
        super().__init__(model=model, hwc=False, **kwargs)
        self.numpyToTensor = NumpyToTensor()
        self.model_pred_fct = model_pred_fct
        self.eval = eval
        allowed_fcts = ['softmax', 'sigmoid', 'linear', 'id', 'tanh', None, False]
        if model_pred_fct not in allowed_fcts:
            print('not allowed model_pred_fct %s' % str(model_pred_fct))
            raise ValueError('model_pred must be in %s' % str(allowed_fcts))
        self._hwc = hwc #self.hwc seems taken, leads to an error in formward if set
        self.avgpool = avgpool
        self.layer = layer
        if layer is None and layer_name is not None:
            self.layer = layer_name
        self.device = None

    def _init_network(self):

        self.device = create_device()

        if self.model is None or is_string_or_path(self.model):
            print('initializing network..')
            model = self._create_model()
            #if done here, problem with torch summary when multiple gpus present, apparently
            # print_network(model)
            if torch.cuda.device_count() > 1:
                print('Using nn.DataParallel for %d gpus' % torch.cuda.device_count())
                model = nn.DataParallel(model)

            model = model.to(self.device)
            print('created model %s on device %s' % (model.__class__.__name__, str(self.device)))
        else:
            model = self.model
            #check if pytorch model has no device or has a different device from self.device
            if not hasattr(self.model, 'device') or self.model.device != self.device:
                model = model.to(self.device)
                # print('put model %s on device %s' % (model.__class__.__name__, str(self.device)))
            else:
                print('using model %s on device %s' % (model.__class__.__name__, str(self.device)))

        if self.layer is not None:
            print('creating hook for layer %s' % str(self.layer))
            self.hook = ModelHook(model, self.layer)

        if self.eval:
            model.eval()
        return model

    def _print_model_summary(self, input_shape=None):
        if torch.cuda.device_count() > 1:
            print('several gpus, not printing model summary! FIXME!')
            return

        df, text = model_summary(self.network, input_shape, device=self.get_device(), forward_fct=self._forward)
        print(text)

    def _forward(self, inp):
        out = self._forward_network(inp)
        if self.layer is not None:
            out = self.hook.out
        return out

    def _forward_network(self, inp):
        out = self.network(inp)
        return out

    def get_device(self):
        if self.device is None:
            self.device = create_device()
        return self.device

    def _encode(self, data):
        data = self.numpyToTensor(data)
        data = data.to(self.get_device())
        if self.calls==1:
            print('first encode call with data %s of type %s using device %s' %\
                  (str(data.shape), str(data.dtype), str(self.get_device())))
        prev = torch.is_grad_enabled()
        if self.eval and prev:
            torch.set_grad_enabled(False)
        repr = self._forward(data)
        if self.model_pred_fct:
            if 'softmax'==self.model_pred_fct:
                repr = F.softmax(repr, dim=1)
            elif 'sigmoid'==self.model_pred_fct:
                repr = torch.sigmoid(repr)
            elif 'tanh'==self.model_pred_fct:
                repr = F.tanh(repr)
            elif self.model_pred_fct in ['linear','id',None,False] :
                pass
            else: raise ValueError('unknown model_pred_fct %s' % self.model_pred_fct)
        else:
            if hasattr(self.network, 'last_representation'):
                repr = self.network.last_representation
        if is_list(repr):
            print('repr list of lenght %d' % len(repr))
            print('shapes:', [str(r.shape) for r in repr])
            repr = repr[0]

        if self._hwc:
            #with pytorch convert from bhwc to bchw
            repr = repr.permute(0,3,1,2)

        if self.avgpool and len(repr.shape)>2:
            repr = F.avg_pool2d(repr, kernel_size=repr.size()[-2:])

        repr = repr.squeeze(-1).squeeze(-1)
        repr = to_numpy(repr)
        if self.eval and prev:
            torch.set_grad_enabled(True)

        if self.calls==1 or self.calls % 1000==0:
            print('GPU max allocated:', max_gpu_mem_allocated(self.get_device(), mb=True))
            sys.stdout.flush()
            sys.stderr.flush()
        return repr

    def _create_model(self):
        raise ValueError('implement _create_model!')

import timm
def create_timm_model(name, pretrained=False, **kwargs):
    model = timm.create_model(name, pretrained=pretrained, **kwargs)
    model.short_name = name
    return model

class TimmEncoder(PytorchEncoderWrapper):
    def __init__(self, model_name, normalizer='imagenet', pretrained=True, **kwargs):
        self.pretrained = pretrained
        super().__init__(model_name=model_name, normalizer=normalizer, **kwargs)

    def _create_model(self):
        import timm
        # enc = timm.create_model(self.model_name, pretrained=True)
        targs = {}
        if self.model is not None and str(self.model)!='None':
            targs['pretrained_cfg_overlay'] = dict(file=self.model)
        enc = create_timm_model(self.model_name, pretrained=self.pretrained, **targs)
        self.n_features = enc.n_features
        return enc

    def _forward_network(self, inp):
        out = self.network.forward_features(inp)
        return out

class LighntningUnsupervisedEncoderWrapper(PytorchEncoderWrapper):
    def __init__(self, name, *args, model_path=None,  **kwargs):
        self.name = name
        super().__init__(*args, **kwargs, model_path=model_path)

    def _create_model(self):
        from pl_bolts.models.self_supervised import SimCLR

        if self.name=='plbolts_simclr':
            weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'
            simclr = SimCLR.load_from_checkpoint(weight_path, strict=False)
            simclr.freeze()
            # model = simclr.encoder
            model = simclr
        else:
            raise ValueError('unknown pl-bolts model %s' % self.name)

        return model


class IncRes2Encoder(TimmEncoder):
    def __init__(self, **kwargs):
        super().__init__(model_name='inception_resnet_v2', layer_name='mixed_7a.branch3_MaxPool2d',
                         avgpool=True, **kwargs)


class Res50Encoder(PytorchEncoderWrapper):
    def __init__(self, last=False, **kwargs):
        if last:
            layer_name = 'layer4.2.relu_ReLU'
        else:
            layer_name = 'layer3.5.relu_ReLU'
        super().__init__(normalizer='imagenet', model_name='res50', layer_name=layer_name, avgpool=True, **kwargs)

    def _create_model(self):
        model = torchvision.models.resnet50(pretrained=True)
        return model


class MtdpEncoderWrapper(PytorchEncoderWrapper):
    def __init__(self, model_path=None, model_pred_fct='id', arch="resnet50", pool=True, **kwargs):
        self.arch = arch
        self.pool = pool
        return super().__init__(normalizer='imagenet', model_path=model_path, model_pred_fct=model_pred_fct, **kwargs)

    def _create_model(self):
        from mtdp import build_model #requires the mtdp code
        model = build_model(arch=self.arch, pretrained="mtdp", pool=self.pool)
        model.eval()
        return model

class DummyEncoder(object):
    def __init__(self, sleep=0):
        self.sleep = sleep

    def __call__(self, data):
        if self.sleep > 0:
            print('dummy sleeping %d seconds...' % self.sleep)
            time.sleep(self.sleep)
        return np.ones((len(data),1), dtype=np.uint8)

    def print_model_summary(self, *args, **kwargs):
        pass


class HistogramEncoder(PatchStatEncoder):
    def __init__(self, n_bins=10, range=(0,1)):
        self.n_bins = n_bins
        self.range = range

    def _encode(self, arr):
        hist = np.histogram(arr, bins=self.n_bins, range=self.range)[0] / len(arr)
        return hist


class DensenetEncoder(PytorchEncoderWrapper):
    def __init__(self, last=False, **kwargs):
        if last:
            layer_name = 'features.norm5_BatchNorm2d'
        else:
            # layer_name = 'layer4.2.relu_ReLU'
            layer_name = 'features.transition3.relu_ReLU'
        super().__init__(normalizer='imagenet', model_name='dense', layer_name=layer_name, avgpool=True, **kwargs)

    def _create_model(self):
        model = torchvision.models.densenet121(pretrained=True)
        return model