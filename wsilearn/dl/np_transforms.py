import random
import numpy as np

import torch

from wsilearn.utils.cool_utils import hwc_to_chw, is_iterable, set_seed

class ImageRangeNormalizer(object):
    """ divides input by 255 if int """
    def __call__(self, obj):
        if obj.dtype=='int':
            return obj/255
        return obj

class NumpyToTensor(object):
    def __init__(self, normalize_image01=False):
        self.normalize_image = normalize_image01

    def __call__(self, obj):
        if min(obj.strides)<0:
            obj = obj.copy()
        if self.normalize_image:
            if 'int' in obj.dtype.name:
                obj = obj.astype(np.float32) / 255
        if not obj.flags['WRITEABLE']:
            obj = obj.copy()  #otherwise pytorch might complain
        return torch.from_numpy(obj)

class RandomFlip(object):

    def __init__(self, horizontal=True, vertical=False, p=0.5, hwc=True):
        """
        Randomly flip an image horizontally and/or vertically with probability p
        p : probability float between [0,1]
        expects an image in format [H,W,C] if hwc
        """
        self.horizontal = horizontal
        self.vertical = vertical
        self.p = p
        self.hwc = hwc

    def __call__(self, x, y=None):
        # x = x.numpy()
        # if y is not None:
        #     y = y.numpy()
        # horizontal flip with p = self.p

        #if C,H,W
        if not self.hwc: #chw->hwc
            x = x.transpose(1,2,0)

        flipped = False
        if self.horizontal:
            if random.random() < self.p:
                x = np.fliplr(x)
                flipped = True
        if self.vertical:
            if random.random() < self.p:
                x = np.flipud(x)
                flipped = True
        if flipped:
            x = np.ascontiguousarray(x)
            # x = x.copy() #necessary, otherwise numpy array non-continguous

        if not self.hwc: #hwc->chw
            x = x.transpose(2,0,1)

        if y is None:
            return x
        return x, y


class RandomRotate(object):
    def __init__(self, p=0.5, hwc=True):
        """
        Randomly rotate  an array with probability p
        p : probability float between [0,1]
        expects an image in format [H,W,C] if hwc
        """
        self.p = p
        self.hwc = hwc
        if self.hwc:
            self.axes = (0,1)
        else:
            self.axes = (1,2)

    def __call__(self, x, y=None):
        angle = np.random.choice([0, 90, 180, 270])
        x = self._rotate(x, angle=angle)
        if y is not None:
            y = self._rotate(y, angle=angle)
        if y is None:
            return x
        return x, y

    def _rotate(self, arr, angle):
        """
        90 degree rotation.
        Args:
            arr: batch in [h,w,c] format.
            angle (int): rotation degree (0, 90, 180 or 270).
            axes: axes to apply the transformation.
        Returns: rotated array.
        """
        axes = self.axes
        if angle == 0:
            pass
        elif angle == 90:
            arr = np.rot90(arr, k=1, axes=axes)
        elif angle == 180:
            arr = np.rot90(arr, k=2, axes=axes)
        elif angle == 270:
            arr = np.rot90(arr, k=3, axes=axes)

        return arr

class Transpose(object):
    def __init__(self, *dims):
        self.dims = dims

    def __call__(self, x, y=None):
        x = x.transpose(*self.dims)
        if y is None:
            return x
        return x, y

class TransposeToChw(object):
    def __init__(self, hwc):
        self.hwc_input = hwc

    def __call__(self, x, y=None):
        if len(x.shape) not in [3,4]: raise ValueError('TransposeToChw expects 3 or 4-dim-array, not %s' % str(x.shape))
        if self.hwc_input:
            x = hwc_to_chw(x)
            # x = x.transpose(2,0,1)
        if y is None:
            return x
        return x, y


class CenterCropPad(object):
    def __init__(self, size, dims=[-2,-1], wiggle=0, **kwargs):
        """ dims: dimension to crop e.g. [1,2] or [-2, -1]
            size: size to crop e.g. 100 or [100,200]
        """
        if not is_iterable(dims):
            dims = [dims]
        if not is_iterable(size):
            size = [size]*len(dims)
        if len(dims)!=len(size):
            raise ValueError('#dims=%d != #size=%d' % (len(dims), len(size)))
        if len(set(dims))!=len(dims):
            raise ValueError('dims have to be unique, not %s' % str(dims))
        self.dims = dims
        self.size = size
        self.wiggle = wiggle
        self.kwargs = kwargs

    def __call__(self, img, center=None, *args, **kwargs):
        n_dims = len(img.shape)
        img_dims = list(range(n_dims))
        #if dims contain neg. values convert to img-relative dims
        dims = [img_dims[d] for d in self.dims]
        # size = [self.size[d] for d in self.dims]
        if center is None:
            center = [self._get_center(img.shape[d], crop_size=self.size[i]) for i,d in enumerate(dims)]

        #pad
        pads = []
        slices = []
        for d in img_dims:
            if d in dims:
                dims_ind = dims.index(d)
                dim = img.shape[d]
                half = self.size[dims_ind]//2
                before = max(0, half-center[dims_ind])
                after = max(0, (self.size[dims_ind]-half) - (dim-center[dims_ind]))
                pads.append((before,after))
                padded_start = before+center[dims_ind]-half
                slices.append(slice(padded_start,padded_start+self.size[dims_ind]))
            else:
                pads.append((0,0))
                slices.append(slice(None))
        if np.max(pads)>0:
            img = np.pad(img, pads, **self.kwargs)
        img = img[tuple(slices)]

        return img

    def _get_center(self, size, crop_size):
        center = size//2
        wiggle = int(max(size-crop_size,0)*self.wiggle*0.5)
        if wiggle:
            shift = random.randint(-wiggle, wiggle)
            center += shift
            center = min(max(0,center),size)

        return center

class RandomCropPad(CenterCropPad):
    def __init__(self, size, dims=[-2,-1], **kwargs):
        super().__init__(size=size, dims=dims, wiggle=1, **kwargs)


class CenterCrop(object):
    """ deprecated, use CenterCropPad"""
    def __init__(self, size):
        self.size = size

    def __call__(self, img, y=None):
        y,x,c = img.shape
        startx = x//2 - self.size//2
        starty = y//2 - self.size//2
        return img[starty:starty+self.size, startx:startx+self.size, :]

class CenterPad(object):
    """ deprecated, use CenterCropPad"""
    def __init__(self, size, dims=[0,1], **kwargs):
        if not is_iterable(dims):
            dims = [dims]
        self.dims = dims
        self.size = size
        self.kwargs = kwargs

    def __call__(self, img, *args, **kwargs):
        n_dims = len(img.shape)
        if is_iterable(self.size):
            size = self.size
        else:
            size = [self.size]*n_dims

        pads = []
        for d in range(n_dims):
            if d in self.dims:
                diff = size[d]-img.shape[d]
                if diff < 0:
                    raise ValueError(f'cant centerpad too large image, padsize: {self.size}, img_size:{img.shape}')
                bef = diff//2
                aft = diff // 2 + diff % 2
                pads.append((bef,aft), **kwargs)
            else:
                pads.append((0,0))
        img = np.pad(img, pads)
        return img

def _test_flip():

    x = np.ones((3, 100, 200))
    x[0] = 0.5
    x[1] = 0.7
    x[2] = 0.9
    x[:,20:40,20:50] = 0

    x = x.transpose(1, 2, 0)
    lr = np.fliplr(x)
    ud = np.flipud(x)
    # x = x.transpose(2, 0, 1)
    #
    # x = x.transpose(1,2,0)
    # lr = lr.transpose(1,2,0)
    # ud = ud.transpose(1,2,0)
    from wsilearn.utils.cool_utils import showx
    showx(x, lr, ud, titles=['Original', 'fliplr', 'flipup'])

def _test_centerpad():
    set_seed(1)
    # size=(12,7)
    size=(13,21)
    x = np.ones((13,7,3))
    #fill diag
    vals = np.linspace(0.3, 0.7, x.shape[1])
    for i in range(min(x.shape[:2])):
        x[i,i]=vals[i]

    pad = CenterCropPad(size=size, dims=[0,1], wiggle=0.5)
    y = pad(x)
    # print(x.shape, y.shape)

    pad2 = CenterCropPad(size=size, dims=[-3, -2])
    y2 = pad2(x)
    from wsilearn.utils.cool_utils import showim, showims
    print(x.shape, y.shape, y2.shape)

    showims(y, y2)


if __name__ == '__main__':
    # _test_flip()
    _test_centerpad()
