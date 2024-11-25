from copy import deepcopy

import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from albumentations import HorizontalFlip, VerticalFlip, Rotate, Compose, ShiftScaleRotate, HueSaturationValue, \
    RandomBrightnessContrast, ImageCompression, GaussianBlur


class AugmentedEncoding(object):
    def __init__(self, encoder):
        self.encoder = encoder

    def encode(self, batch):
        encoded = self.encoder(batch)
        return encoded

    def __call__(self, batch):
        return self.encode(batch)


class MeanAugmentedEncoding(AugmentedEncoding):
    def __init__(self, encoder, transforms):
        super().__init__(encoder)
        self.transforms = transforms

    def encode(self, batch):
        encoded = [self.encoder(batch)]
        for transf in self.transforms:
            batch_aug = []
            for patch in batch:
                augm = transf(image=patch)['image']
                batch_aug.append(augm)
            batch_aug = np.stack(batch_aug)
            batch_aug_enc = self.encoder(batch_aug)
            encoded.append(batch_aug_enc)
        encoded = np.stack(encoded)
        encoded_mean = encoded.mean(0)
        return encoded_mean


class RotationMeanEncoding(MeanAugmentedEncoding):
    def __init__(self, encoder):
        transforms = [HorizontalFlip(p=1), VerticalFlip(p=1),
                      Rotate((90,90),p=1), Rotate((180,180),p=1), Rotate((270,270),p=1),
                      Compose([Rotate((90,90),p=1),VerticalFlip(p=1)]),
                      Compose([Rotate((270,270),p=1),VerticalFlip(p=1)])
                    ]
        super().__init__(encoder, transforms)


class RandomAugmentationEncoding(AugmentedEncoding):
    def __init__(self, encoder, transform):
        super().__init__(encoder)
        self.transform = transform

    def encode(self, batch):
        augmented = []
        for patch in batch:
            augm = self.transform(image=patch)['image']
            augmented.append(augm)
        augmented = np.stack(augmented)
        encoded = self.encoder(augmented)
        return encoded


class RandomAugmentationEncoding1(RandomAugmentationEncoding):
    def __init__(self, encoder):
        transforms = [HorizontalFlip(p=0.1),VerticalFlip(p=0.1),
                      Rotate((90,90),p=0.1), Rotate((180,180),p=0.1), Rotate((270,270),p=0.1),
                      ShiftScaleRotate(shift_limit=0.03125, scale_limit=0.05, rotate_limit=0, p=0.25),
                      HueSaturationValue(15, 20, 20, p=0.25),
                      RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.25),
                      # GaussianBlur(blur_limit=(3,5), p=0.1),
                      ImageCompression(75, 85, p=0.25),
                      ]
        transforms = Compose(transforms)
        super().__init__(encoder, transforms)


class ColorMeanEncoding(MeanAugmentedEncoding):
    def __init__(self, encoder):
        augmentations = [
            ShiftScaleRotate(shift_limit=0.03125, scale_limit=0.05, rotate_limit=0, p=0.5),
            HueSaturationValue(15, 20, 20, p=0.5),
            RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            GaussianBlur(blur_limit=(3,5), p=0.2),
            ImageCompression(70, 85, p=0.5),
        ]

        transforms = []
        for i in range(8):
            transform = Compose(deepcopy(augmentations))
            transforms.append(transform)

        super().__init__(encoder, transforms)


class RotationColorMeanEncoding(MeanAugmentedEncoding):
    def __init__(self, encoder, n=4):
        augmentations = [
            HorizontalFlip(p=0.1),VerticalFlip(p=0.1),
            Rotate((90,90),p=0.1), Rotate((180,180),p=0.1), Rotate((270,270),p=0.1),
            ShiftScaleRotate(shift_limit=0.03125, scale_limit=0.05, rotate_limit=0, p=0.5),
            HueSaturationValue(15, 20, 20, p=0.5),
            RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            GaussianBlur(blur_limit=(3,5), p=0.2),
            ImageCompression(70, 85, p=0.5),
        ]

        transforms = []
        for i in range(n):
            transform = Compose(deepcopy(augmentations))
            transforms.append(transform)

        super().__init__(encoder, transforms)


patch_encoding_map = {'none': AugmentedEncoding, 'rot8m': RotationMeanEncoding, 'color8m': ColorMeanEncoding,
                      'raug1': RandomAugmentationEncoding1, 'rc4m': RotationColorMeanEncoding}