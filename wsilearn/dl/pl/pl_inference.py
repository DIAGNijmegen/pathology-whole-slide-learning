from abc import abstractmethod
from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from copy import deepcopy
import torch.nn.functional as F
from torch import autocast
from torch.cuda.amp import GradScaler

from wsilearn.utils.cool_utils import is_dict, is_iterable, dict_of_lists_to_dicts, is_ndarray, mkdir, time_to_str, \
    save_arrays, is_callable
from wsilearn.dl.torch_utils import to_numpy, create_device
from time import time

from wsilearn.utils.path_utils import PathUtils

_default_data_key = 'data'
_default_target_key = 'target'
_default_out_key = 'out'
_default_logit_key = 'logit'

def torch_to_python(obj):
    if torch.is_tensor(obj):
        return to_numpy(obj)
    else:
        return obj

def inference_trainer(trainer, module, dataloader, ckpt_path='best', data_key=_default_data_key):

    print('inference %d samples' % len(dataloader))
    entries = []
    for data in dataloader:
        dkeys = list(data.keys())
        other_keys = [k for k in data if k not in [data_key]]
        d_list = dict_of_lists_to_dicts(data)
        for d in d_list:
            entries.append({k:torch_to_python(d[k]) for k in other_keys})
    preds = trainer.predict(module, ckpt_path=ckpt_path, dataloaders=dataloader)
    preds = torch.concat(preds)
    preds = to_numpy(preds)
    return preds, entries
    # preds = probs.argmax(1)
    # acc = accuracy_score(targets, preds)
    # print('sk acc: %.8f (%d samples)' % (acc, len(preds)))

def load_pl_state_dict(path, model, replacements_pl={'model.':''}, replacements={}, deletions=[], **kwargs):
    """ Loads the state dict of the pl checkpoint into the model. removes pl-specific weight-name prefixes"""
    if not torch.cuda.is_available():
        kwargs['map_location'] = 'cpu'
    state_dict = torch.load(path, **kwargs)['state_dict']
    # for key in list(state_dict.keys()):
    #     if key.startswith('model.'):
    #         state_dict[key.replace('model.', '')] = state_dict.pop(key)
    #     else:
    #         state_dict.pop(key) #ignore extra modules

    if 'criterion.weight' not in deletions:
        deletions.append('criterion.weight')
    replacements_pl.update(replacements)
    for key in list(state_dict.keys()):
        deleted = False
        for k in deletions:
            if key.startswith(k):
                state_dict.pop(key)
                deleted = True
                break
        if deleted: continue
        for k,v in replacements_pl.items():
            if k in key:
                state_dict[key.replace(k, v)] = state_dict.pop(key)

    model.load_state_dict(state_dict) #has 'model.' prefix

def find_pl_model_path(out_dir, last=False):
    """ finds the best or last (if flag set) model path """
    if last:
        last_path = Path(out_dir)/'last.ckpt'
        if not last_path.exists():
            raise ValueError('last model not found in %s' % str(out_dir))
        return last_path
    else:
        ckpts = PathUtils.list_pathes(out_dir, ending='ckpt', containing='best')
        if len(ckpts)==0: raise ValueError('not best model path found in %s' % str(out_dir))
        if len(ckpts)>1:
            print('warning: %d best model pathes found: %s' % (len(ckpts), str(ckpts)))
        return str(ckpts[0])

class InferenceCallback(object):
    def __init__(self):
        self.time = 0
        self.disabled = False

    def pre(self, batch, x, data, out_dir):
        if self.disabled: return
        start_time = time()
        self._pre(batch=batch, x=x, data=data, out_dir=out_dir)
        self.time+= time()-start_time

    @abstractmethod
    def _pre(self, batch, x, data, out_dir):
        pass

    def post(self, batch, x, data, out, out_dir):
        if self.disabled: return
        start_time = time()
        self._post(batch, x, data, out, out_dir)
        self.time+= time()-start_time

    @abstractmethod
    def _post(self, batch, x, data, out, out_dir):
        pass

    def print_time(self):
        time_str = self.__class__.__name__+' time: '+time_to_str(self.time)
        print(time_str)
        return time_str

""" Writes the output as file e.g. npz """
class InferenceOutWriter(InferenceCallback):
    def __init__(self, overwrite=False, format='npz', name_col='name', out_dir_name=None,
                 default_out_col='out'):
        """ name_col: used to retrieve the name of the file from the data-dict
         default_out_col: key to save the output if it is just a tensor
         """
        super().__init__()
        self.overwrite = overwrite
        self.format = format
        self.time = 0
        self.name_col = name_col
        self.out_dir_name = out_dir_name
        self.default_out_col = default_out_col

        allowed_formats = ['npz']
        if format not in allowed_formats:
            raise ValueError('only formats %s allowed, not %s' % (str(allowed_formats), format))

    def _post(self, batch, x, data, out, out_dir):
        if self.out_dir_name is not None:
            out_dir = Path(out_dir)/self.out_dir_name
        else:
            out_dir = Path(out_dir)
        mkdir(out_dir)
        # data = {k:v.item() for k,v in data.items()}
        if not is_dict(out):
            out = {self.default_out_col:out}
        out_dicts = {}
        for k,v in out.items():
            if torch.is_tensor(v):
                v = to_numpy(v)
            out_dicts[k] = v
        out_dicts = dict_of_lists_to_dicts(out_dicts)
        data_dicts = dict_of_lists_to_dicts(data)
        for b,data in enumerate(data_dicts):
            name = data[self.name_col]
            out_path = Path(out_dir)/('%s.%s' % (name, self.format))
            if out_path.exists() and not self.overwrite:
                print('skip existing %s' % str(out_path))
                continue
            outb = out_dicts[b]
            self._save(data, outb, out_path)

    def _save(self, data, out, out_path):
        if self.format=='npz':
            save_arrays(out_path, **out, **data)

class Inferencer(object):
    def __init__(self, model, callbacks=[], device=None, data_key=_default_data_key, out_key=_default_out_key,
                 logit_key=_default_logit_key, overwrite=False, post_fct=None, fp16=False, batch_is_sample=False,
                 benchmark=True):
        """ batch_is_sample: if True, the batch is a single sample, not a batch of samples """
        if callbacks is None:
            callbacks = []
        self.callbacks = callbacks

        if device is None:
            device = create_device(benchmark=benchmark)
        self.device = device

        model.eval()
        model = model.to(device)

        self.model = model

        self.data_key = data_key
        self.out_key = out_key
        self.logit_key = logit_key

        self.overwrite = overwrite

        self.batch_is_sample = batch_is_sample

        post_fcts = [None, 'id', 'sigmoid', 'softmax','surv']
        if post_fct not in post_fcts and not is_callable(post_fct):
            raise ValueError('post_fct %s not in %s' % (post_fct, str(post_fcts)))
        self.post_fct = post_fct
        self.fp16 = fp16

        self._warned = False

    def _post_process(self, output):
        if is_dict(output):
            out = output[self.out_key]
        else:
            out = output
        if self.post_fct:
            if is_callable(self.post_fct):
                out = self.post_fct(out)
            elif 'softmax'==self.post_fct:
                if out.shape[1]==1:
                    out = torch.sigmoid(out)
                    if not self._warned:
                        print('CHANGED SOFTMAX TO SIGMOID')
                        self._warned = True
                else:
                    out = F.softmax(out, dim=1)
            elif 'sigmoid'==self.post_fct:
                out = torch.sigmoid(out)
            elif 'tanh'==self.post_fct:
                out = F.tanh(out)
            elif 'surv'==self.post_fct:
                if out.shape[1]==1: #coxph
                    out = torch.exp(-out)
                else: #nll and my multisurv
                    out = torch.sigmoid(out)
            elif self.model_pred_fct in ['linear','id',None,False] :
                pass
            else:
                raise ValueError('unknown post_fct %s' % self.post_fct)
        if is_dict(output):
            output[self.out_key] = out
            return output
        else:
            return out

    def apply(self, loader, out_dir):
        if out_dir is not None:
            out_path = Path(out_dir)/'out.csv'
            if out_path.exists() and not self.overwrite:
                print('skipping applying due to existing %s' % str(out_path))
                df = pd.read_csv(out_path)
                return df

        assert self.model.training==False #set to eval in init

        print('applying to %d batches, out=%s' % (len(loader), str(out_dir)))
        entries = []
        with torch.no_grad():
            for batch, data in enumerate(loader):
                if not is_dict(data):
                    data = {self.data_key:data}
                x = data.pop(self.data_key)
                for cb in self.callbacks:
                    cb.pre(batch, x, data, out_dir=out_dir)
                if x.dtype == torch.float16:# and not self.fp16:
                    x = x.float()
                x = x.to(self.device)

                bs = len(x)
                try:
                    # with autocast(self.device.type, enabled=self.fp16): #doest seem to make a difference
                    out = self.model(x)
                except:
                    print('rest-data:', data)
                    raise

                if is_dict(out):
                    logits = out[self.out_key]
                else:
                    logits = out
                out = self._post_process(out)

                for cb in self.callbacks:
                    cb.post(batch, x, data, out, out_dir=out_dir)

                if is_dict(out):
                    out = out[self.out_key]

                out = to_numpy(out)
                logits = to_numpy(logits)
                #add results

                self._add_result_entries(data, logits, out, entries)

        df = pd.DataFrame(entries)
        if out_dir is not None:
            mkdir(out_dir)
            df.to_csv(str(out_path), index=False)
        return df
        #     probs.append(to_numpy(prob))
        #     targets.append(to_numpy(y))
        # probs = np.concatenate(probs).argmax(1)
        # targets = np.concatenate(targets)
        # acc = accuracy_score(targets, probs)
        # print('sk acc: %.8f' % acc)

    def _add_result_entries(self, data, logit, out, entries):
        bs = len(out)
        if len(data) == 0:
            data_entries = [{}] * bs
        else:
            if self.batch_is_sample:
                n_out = len(out)
                if n_out !=1: raise ValueError('batch_is_sample=True but len(out)=%d' % n_out)
                data = deepcopy(data)
                for k in list(data.keys()):
                    kvals = list(set(data[k]))
                    if len(kvals)==1:
                        data[k] = [kvals[0]]
                    else:
                        data.pop(k)
            data_entries = dict_of_lists_to_dicts(data)
        for bind in range(bs):
            entry = {}
            entries.append(entry)
            out_bind = out[bind]
            logit_bind = logit[bind]
            for j in range(len(out_bind)):
                entry[self.out_key + str(j)] = out_bind[j]
            for j in range(len(out_bind)):
                entry[self.logit_key + str(j)] = logit_bind[j]

            n_out = len(out)
            for k, v in data_entries[bind].items():
                if torch.is_tensor(v):
                    v = to_numpy(v)
                if is_ndarray(v) and len(v.shape) == 0:  # unsized object array
                    v = v.item()
                if is_iterable(v):
                    if len(v) <= n_out:  # ignore longer entries
                        for j in range(len(v)):
                            entry[k + str(j)] = v[j]
                else:
                    entry[k] = v

        pass