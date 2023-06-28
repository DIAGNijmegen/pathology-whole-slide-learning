import torch
from torch import nn
import torch.nn
import torch.nn.functional as F

class GlobalAvgPool2d(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[-2:])

class GlobalMaxPool2d(nn.Module):
    def forward(self, x):
        return F.max_pool2d(x, kernel_size=x.size()[-2:])

class MaxAvgPool2d(nn.Module):
    def __init__(self, *args, weighted=False, **kwargs):
        super().__init__()
        self.maxpool = nn.MaxPool2d(*args, **kwargs)
        self.avgpool = nn.AvgPool2d(*args, **kwargs)
        if weighted:
            self.alpha = torch.nn.Parameter(torch.tensor(0.5), requires_grad=True)
        else:
            self.alpha = 0.5
        self.kernel_size = self.maxpool.kernel_size
        self.stride = self.maxpool.stride

    def forward(self, x):
        dim = 1
        if len(x.shape)==3: #no batch dimension
            dim = 0
        return torch.cat([self.alpha*self.maxpool(x), (1-self.alpha)*self.avgpool(x)], dim)


class GlobalGemPool2d(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)
        return F.avg_pool2d(x.clamp(min=eps).pow(p), kernel_size=x.size()[-2:]).pow(1. / p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'

def create_pool(name='maxpool', kernel_size=(2,2), stride=2, **kwargs):
    if name=='maxpool' or name=='max_pool' or name=='max':
        return nn.MaxPool2d(kernel_size=kernel_size, stride=stride, **kwargs)
    elif name=='avgpool' or name=='avg_pool' or name=='avg':
        return nn.AvgPool2d(kernel_size=kernel_size, stride=stride, **kwargs)
    elif name.lower() == 'max_avg_pool' or name.lower() == 'max_avg' or name.lower()=='maxavg':
        return MaxAvgPool2d(kernel_size=kernel_size, stride=stride)

    elif name.lower() == 'global_avg' or name.lower() == 'global_avg_pool' or name.lower() == 'global_avgpool' or name.lower() == 'gavg':
        return GlobalAvgPool2d()
    elif name.lower() == 'global_max_pool' or name.lower() == 'global_maxpool' or name.lower() == 'global_max' or name.lower() == 'gmax':
        return GlobalMaxPool2d()
    elif name.lower()=='gem':
        return GlobalGemPool2d()
    else:
        raise ValueError('unknown pool operation %s' % name)

def nms_set(tensor, kernel_size=3, val=0, **kwargs):
    if torch.is_tensor(val) and val.requires_grad:
        val = val.detach()
    padding=(kernel_size-1)//2
    pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1,  padding=padding,
                        return_indices=True, **kwargs)
    ind_arr = torch.arange(0, tensor.numel()).reshape(tensor.shape).to(tensor.device)
    last_peaks = torch.zeros_like(tensor)
    #iteratively determine the peaks until nothing changes or the timer runs out
    for i in range(10):
        # print('tensor:')
        # print(input)
        pooled, indices = pool(tensor)
        peaks = ind_arr==indices
        peaks = peaks.type(torch.float)
        # print('peaks:')
        # print(peaks)

        dilated = F.max_pool2d(peaks, kernel_size=kernel_size, stride=1, padding=padding)
        # print('dilated')
        # print(dilated)

        nms = dilated-peaks
        # print('nms')
        # print(nms)

        tensor[nms==1] = val
        if (last_peaks==peaks).all():
            # print('stopping at iteration %d' % i)
            break
        last_peaks = peaks
    return tensor