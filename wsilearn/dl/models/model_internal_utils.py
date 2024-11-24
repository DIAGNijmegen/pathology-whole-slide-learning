import torch
import torch.nn as nn

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

class ModelHook(object):
    def __init__(self, model, layer_name):
        self.out = None
        self.inp = None

        name_module_map = get_module_names_dict(model)
        counter = 0
        for name, module in name_module_map.items():
            if name==layer_name:
                module.register_forward_hook(self._forward_hook)
                counter+=1
        if counter==0:
            raise ValueError('layer %s not found, available layers: %s' % (layer_name, str(list(name_module_map.keys()))))
        if counter>1:
            raise ValueError('%d layers with name %s found' % (counter, layer_name))


    def _forward_hook(self, module, inp, out):
        self.inp = inp
        if len(out.shape)==3: #vit, #fixme: this is a hack
            out = out[:, 0] #prediction token
        self.out = out


def register_forward_hook(net, hook_fn, name=None):
    for lname, layer in net._modules.items():
        # for sequential recursively register hook on all it's module children
        if isinstance(layer, nn.Sequential):
            register_forward_hook(layer, hook_fn)
        else:
            reg = True
            if name is not None and name!=lname:
                reg = False
            if reg:
                layer.register_forward_hook(hook_fn)
