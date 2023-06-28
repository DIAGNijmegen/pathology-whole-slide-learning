import numpy as np

class ImageNormalizer255To1(object):
    def __call__(self, patch):
        if patch.dtype!=np.uint8:
            raise ValueError('expects np.uint8, not', patch.dtype)
        return (patch / 255.0).astype(np.float32)

    def reverse(self, patch):
        patch = np.clip(patch* 255 + 0.5, 0, 255).astype(np.uint8)
        return patch

class ImageNormalizerMinusOneToOne(object):
    def __call__(self, patch):
        return (((patch / 255.0) * 2)-1).astype(np.float32)

    def reverse(self, patch):
        patch = np.clip(((patch+1)*255/2.0)+0.5,0, 255).astype(np.uint8)
        return patch

def get_imagenet_mean_std():
    return np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225], dtype=np.float32)

class MeanStdImageNormalizerNp(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        if img.dtype==np.uint8:
            img = img/255.0
        if img.dtype==np.float64: #this is to make it more similar to the pytorch version regarding rounding errors (probably)
            img = img.astype(np.float32)

        result = (img-self.mean)/self.std
        return result.astype(np.float32)

class ImagenetNormalizerNp(MeanStdImageNormalizerNp):
    def __init__(self):
        super().__init__(*get_imagenet_mean_std())

class HalfNormalizerNp(MeanStdImageNormalizerNp):
    def __init__(self):
        super().__init__(np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5], dtype=np.float32))

def create_input_normalizer(name):
    if name=="01" or name=="rgb_to_0-1":
        return ImageNormalizer255To1()
    elif name=="-11":
        return ImageNormalizerMinusOneToOne()
    elif name=='imagenet':
        return ImagenetNormalizerNp()
    elif name in ['half','histossl','hipt']:
        return HalfNormalizerNp()
    else:
        raise ValueError('Unknnown input normalization %s' % name)