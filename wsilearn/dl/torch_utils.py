from pathlib import Path

import functools

from wsilearn.utils.df_utils import print_df

print = functools.partial(print, flush=True)

from wsilearn.utils.cool_utils import is_ndarray, CaptureOut, is_list_or_tuple, set_seed, is_string, is_callable, \
    is_iterable, write_lines, write_text
import torch
from torch import nn
import math
from copy import deepcopy
import torch.backends.cudnn as cudnn
import numpy as np

def set_seed_all(seed, cuda=True):
    print('setting seed %d' % seed)
    set_seed(seed)
    torch.manual_seed(seed)
    if cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def create_device(benchmark=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    if device.type == 'cuda' and benchmark:
        cudnn.benchmark = True
    if str(device)=='cpu':
        name = 'cpu'
    else:
        name = torch.cuda.get_device_name(device)
    print(f'{device.type} device {device.index}: {name}')
    return device


def get_module_names_dict(model):
    """Recursive walk to get names including path."""
    names = {}

    def _get_names(module, parent_name=""):
        for key, m in module.named_children():
            cls_name = str(m.__class__).split(".")[-1].split("'")[0]
            num_named_children = len(list(m.named_children()))
            if num_named_children > 0:
                name = parent_name + "." + key if parent_name else key
            else:
                mname = key + "_" +cls_name
                name = parent_name + "." + mname if parent_name else mname
            names[name] = m

            if isinstance(m, torch.nn.Module):
                _get_names(m, parent_name=name)

    _get_names(model)
    return names

def to_numpy(var):
    if is_ndarray(var):
        return var
    var = var.detach().cpu()
    if var.dtype==torch.bfloat16:
        var = var.float()
    var = var.numpy()
    if is_iterable(var) and len (var.shape)==0: #0-sized array
        var = var.item()
    return var

def to_cpu(*tensors):
    results = [tensor.detach().cpu() for tensor in tensors]
    if len(results)==1:
        return results[0]
    return results

def max_gpu_mem_allocated(device, mb=False):
    bytes = torch.cuda.max_memory_allocated(device=device)
    if mb:
        bytes = bytes/1000000
    return bytes

def determine_max_input_volume(model, start_size, device=None, reserve=0.1, train=True, even_batch_size=True, loss=False):
    if device is None:
        device = create_device(benchmark=False)
    print('determine_max_input_volume...')
    print("CUDA memory allocated:", torch.cuda.memory_allocated())
    m = deepcopy(model)
    m = m.to(device)
    if train:
        m.train()
    else:
        if loss: raise ValueError('loss doesnt make sense with train=False')
        m.eval()
    if len(start_size)==3:
        start_size = start_size[None,:,:,:]
    else:
        start_size[0] = 1

    if train and loss:
        optimizer = torch.optim.Adam(m.parameters(), lr=1e-4, weight_decay=1e-4)
        criterion = nn.MSELoss()

    working_size = None
    print('determine_max_input_volume, starting with size %s' % str(start_size))
    factor = 2
    last_try_batch_size = start_size[0]
    gen = torch.Generator()
    while True:
        new_batch_size = int(math.ceil(last_try_batch_size*factor))
        diff = new_batch_size - last_try_batch_size
        if working_size is not None and diff < 0.01*last_try_batch_size:
            print(f'stopping with {working_size} since difference <1perc')
            break
        if even_batch_size and new_batch_size%2 != 0:
            new_batch_size+=1
        try_size = [new_batch_size,*start_size[1:]]
        print(f'trying {try_size}')
        m.zero_grad()
        # start_size = (b+1,*start_size[1:])
        inp = None; out = None
        try:
            inp = torch.randn(try_size, generator=gen).to(device)
            if train:
                out = m(inp)
                if loss:
                    optimizer.zero_grad()
                    dummy_target = torch.randn(out.size(), generator=gen).to(device)
                    # loss = (out - dummy_target).pow(2).sum()
                    loss = criterion(out, dummy_target)
                    loss.backward()
                    optimizer.step()
            else:
                with torch.no_grad():
                    out = m(inp)
            working_size = try_size
            last_try_batch_size = try_size[0]
            m.zero_grad()
            # del inp
            # del out
            # if loss:
            #     del loss
            #     loss = True
            inp = None
            out = None
            loss = False
            torch.cuda.empty_cache()

            print('working %s, max gpu so far: %s' % (str(try_size), max_gpu_mem_allocated(device, mb=True)))

        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                # print('|out of mem at size %s' % str(try_size))
                # traceback.print_exc()

                inp = None
                out = None
                loss = True if loss else False
                ## doesnt work, only deleting the model works
                # m.zero_grad()
                # for p in m.parameters():
                #     if p.grad is not None:
                #         del p.grad  # free some memory
                # del m

                if last_try_batch_size >= try_size[0]-1:
                    print(f'stop with try_size {try_size[0]}')
                    break
                else:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize(device)
                    factor = np.sqrt(factor)
                    print('reducing try search factor to %.5f' % factor)

                    # m = deepcopy(model)
                    # m = m.to(device)
                    # if train:
                    #     m.train()
                    # else:
                    #     if loss: raise ValueError('loss doesnt make sense with train=False')
                    #     m.eval()
            else:
                raise e

    inp = None
    out = None
    loss = None
    del m
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    if working_size is None:
        raise ValueError('determining input volume failed: initial size %s already too large' % str(try_size))
    vol = np.prod(working_size)
    margin_vol = int(vol * (1 - reserve))
    margin_batch_size = int(working_size[0] * (1 - reserve))
    if even_batch_size and margin_batch_size>3 and margin_batch_size%2!=0:
        margin_batch_size = margin_batch_size-1
    assert margin_batch_size > 0
    margin_size = working_size.copy()
    margin_size[0] = margin_batch_size
    print('working size: %s, volume: %d, ~margin_size: %s, volume-margin: %d' %\
          (str(working_size), vol, str(margin_size), margin_vol))
    print("CUDA memory allocated:", torch.cuda.memory_allocated())
    print('determine_max_input_volume finished')
    return margin_size, margin_vol

#adapted from torchsummaryx
#no mult-adds

import io
from contextlib import redirect_stdout
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch

def mysummary(model, x, print_text=False, forward_fct=None, *args, **kwargs):
    """Summarize the given input model.
    Summarized information are 1) output shape, 2) kernel shape,
    3) number of the parameters and 4) operations (Mult-Adds)
    Args:
        model (Module): Model to summarize
        x (Tensor): Input tensor of the model with [N, C, H, W] shape
                    dtype and device have to match to the model
        args, kwargs: Other argument used in `model.forward` function
    """
    def register_hook(module):
        def hook(module, inputs, outputs):
            cls_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            key = None
            # Lookup name in a dict that includes parents
            for name, item in module_names.items():
                if item == module:
                    key = "{}_{}".format(module_idx, name)
            if key is None:
                # print('mysummary.register_hook error: module %s not found in %s',
                #       (str(module), str(list(module_names.keys()))))
                print('mysummary.register_hook error: module %s not found' % str(module))
                # return

            info = OrderedDict()
            info["id"] = id(module)
            if isinstance(outputs, (list, tuple)):
                try:
                    info["out"] = list(outputs[0].size())
                except AttributeError:
                    # pack_padded_seq and pad_packed_seq store feature into data attribute
                    info["out"] = list(outputs[0].data.size())
            else:
                info["out"] = list(outputs.size())

            info["ksize"] = "-"
            info["inner"] = OrderedDict()
            info["params_nt"], info["params"], info["macs"] = 0, 0, 0
            for name, param in module.named_parameters():
                info["params"] += param.nelement() * param.requires_grad
                info["params_nt"] += param.nelement() * (not param.requires_grad)

                if name == "weight":
                    ksize = list(param.size())
                    # to make [in_shape, out_shape, ksize, ksize]
                    if len(ksize) > 1:
                        ksize[0], ksize[1] = ksize[1], ksize[0]
                    info["ksize"] = ksize

                    # ignore N, C when calculate Mult-Adds in ConvNd
                    if "Conv" in cls_name:
                        info["macs"] += int(param.nelement() * np.prod(info["out"][2:]))
                    else:
                        info["macs"] += param.nelement()

                # RNN modules have inner weights such as weight_ih_l0
                elif "weight" in name:
                    info["inner"][name] = list(param.size())
                    info["macs"] += param.nelement()

            # if the current module is already-used, mark as "(recursive)"
            # check if this module has params
            if list(module.named_parameters()):
                for v in summary.values():
                    if info["id"] == v["id"]:
                        info["params"] = "(recursive)"

            if info["params"] == 0:
                info["params"], info["macs"] = "-", "-"

            if key is not None:
                summary[key] = info

        # ignore Sequential and ModuleList
        if not module._modules:
            hooks.append(module.register_forward_hook(hook))

    module_names = get_module_names_dict(model)

    hooks = []
    summary = OrderedDict()

    model.apply(register_hook)
    if forward_fct is None: forward_fct = model
    try:
        with torch.no_grad():
            forward_fct(x) if not (kwargs or args) else forward_fct(x, *args, **kwargs)
    finally:
        for hook in hooks:
            hook.remove()

    if len(summary)==0:
        print('summary failed, no entries')
        return None, ''

    # Use pandas to align the columns
    df = pd.DataFrame(summary).T
    # df0 = pd.DataFrame(inp_layer_dict)
    # df.insert(0, inp_layer_dict)
    # df["Mult-Adds"] = pd.to_numeric(df["macs"], errors="coerce")
    df["Params"] = pd.to_numeric(df["params"], errors="coerce")
    df["Non-trainable params"] = pd.to_numeric(df["params_nt"], errors="coerce")
    df = df.rename(columns=dict(
        ksize="Kernel Shape",
        out="Output Shape",
    ))
    df_sum = df.sum(numeric_only=True)
    n_convs = df.index.str.contains('Conv2d').sum()
    df.index.name = "Layer"


    # df = df[["Kernel Shape", "Output Shape", "Params", "Mult-Adds"]]
    df = df[["Kernel Shape", "Output Shape", "Params"]]
    max_repr_width = max([len(row) for row in df.to_string().split("\n")])

    inp_layer_dict = {'Layer':'Input', 'Kernel Shape': '-', 'Output Shape': list(x.shape), 'Params':'-'}
    df_input = pd.DataFrame([inp_layer_dict])
    df_input.set_index('Layer', inplace=True)
    # df = df_input.append(df)
    df = pd.concat((df_input, df))

    option = pd.option_context(
        "display.max_rows", 1000,
        "display.width", 1000,
        "display.max_columns", 20,
        'max_colwidth', 200,
        "display.float_format", pd.io.formats.format.EngFormatter(use_eng_prefix=True)
    )
    with option:
        # lines = []
        with CaptureOut() as lines:
            print("="*max_repr_width)
            print(df.replace(np.nan, "-"))
            print("-"*max_repr_width)
            df_total = pd.DataFrame([{"Name": "Total params", "Totals": df_sum["Params"] + df_sum["Non-trainable params"]},
                                     {"Name":"Trainable params","Totals": df_sum["Params"]},
                                     {"Name":"Non-trainable params","Totals": df_sum["Non-trainable params"]},
                                     {"Name":"Conv Layers", "Totals": n_convs}
                                     #"Mult-Adds": df_sum["Mult-Adds"]
                                    ])
            df_total.set_index("Name", inplace=True)
            # df_total.Totals = df_total.Totals.astype(np.int)
            print(df_total)
            print("="*max_repr_width)
        text = "\n".join(lines)
        if print_text:
            print(text)

    return df, text

def model_summary(model, inp, device=None, out_path=None, forward_fct=None):
    if is_ndarray(inp):
        inp = torch.tensor(inp)
    elif is_list_or_tuple(inp) or 'tensor' not in inp.type().lower():
        input_shape = inp
        if len(inp)==3:
            input_shape = [2]+list(input_shape)
        inp = torch.zeros(input_shape)
        if len(input_shape)==4:
            inp[0,0,0,0] = 1
    if device is not None:
        inp = inp.to(device)
    model_training = model.training
    if model_training:
        model.eval()
    df, text = mysummary(model, inp, forward_fct=forward_fct)
    # with CaptureOut() as printed_lines:
    #     summary(model, inp)  # Here printing (also when debugging) is going to the capture-string, not stdout
    if model_training:
        model.train()
    if out_path is not None:
        Path(out_path).parent.mkdir(exist_ok=True)
        write_text(out_path, text, overwrite=True)
    return df, text

def print_model_summary(model, inp, **kwargs):
    df, text = model_summary(model, inp, **kwargs)
    print(text)




if __name__ == '__main__':
    pass