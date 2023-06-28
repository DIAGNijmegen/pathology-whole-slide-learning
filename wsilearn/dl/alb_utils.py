import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2

from wsilearn.utils.cool_utils import is_ndarray, is_dict

imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)
def create_norm_transform():
    return A.Normalize(mean=imagenet_mean, std=imagenet_std)

def denormalize_tensor(inp):
    """Convert an imagenet normalized Tensor to denormalized numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array(imagenet_mean)
    std = np.array(imagenet_std)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

def hsv_lighter(size=256, normalize=False, tensor=False):
    transforms = [
        A.ElasticTransform(p=0.1, sigma=10, alpha_affine=20),
        A.RandomResizedCrop(size, size, scale=(0.9, 1.0), ratio=(0.9, 1.1), p=1),
        A.RandomRotate90(p=1),
        A.HorizontalFlip(p=0.2),
        A.VerticalFlip(p=0.2),
        ### A.RandomScale(p=0.5, scale_limit=0.2),
        A.ColorJitter(p=1, brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        A.ImageCompression(p=0.2, quality_lower=70),
        A.GaussNoise(p=0.2),
        A.GaussianBlur(p=0.2, sigma_limit=1)
    ]
    if normalize:
        transforms.append(create_norm_transform())
    if tensor:
        transforms.append(ToTensorV2())

    return A.Compose(transforms)

class SimpleAlbWrapper(object):
    def __init__(self, albs):
        self.albs = albs

    def transform(self, data):
        if not is_dict(data):
            data = dict(image=np.array(data))
        augm = self.albs(**data)
        img = augm['image']
        return img

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

class AlbTransform(object):
    def __init__(self, transform=None):
        self.transf = transform

    def transform(self, data):
        if self.transf is None:
            return data
        augm = self.transform(**data)
        return augm

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)