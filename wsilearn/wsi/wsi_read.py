import os
from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import functools


from wsilearn.utils.cool_utils import timer, showims, multiproc_pool2, showim
from wsilearn.utils.df_utils import print_df
from wsilearn.utils.flexparse import FlexArgumentParser
from wsilearn.utils.path_utils import PathUtils
from wsilearn.wsi.wsi_utils import delete_slide, copy_slide

##################################################################################
############### THIS IS OBSOLETE, SWITCH TO WHOLESLIDEDATA #######################
##################################################################################

print = functools.partial(print, flush=True)
from matplotlib import pyplot as plt
from PIL import Image

try:
    from multiresolutionimageinterface import (
        MultiResolutionImageReader,
        MultiResolutionImage,
    )
except ImportError:
    MultiResolutionImageReader = None
    MultiResolutionImage = None
    print("cannot import MultiResolutionImage")


try:
    from openslide import OpenSlide
except ImportError:
    OpenSlide = None
    print("cannot import OpenSLide")


class InvalidSpacingError(ValueError):
    def __init__(self, image_path, spacing, spacings, margin):

        super().__init__(
            f"Image: '{image_path}', with available pixels spacings: {spacings}, does not contain a level corresponding to a pixel spacing of {spacing} +- {margin}"
        )

        self._image_path = image_path
        self._spacing = spacing
        self._spacings = spacings
        self._margin = margin

    def __reduce__(self):
        return (
            InvalidSpacingError,
            (self._image_path, self._spacing, self._spacings, self._margin),
        )


class UnsupportedVendorError(KeyError):
    def __init__(self, image_path, properties):

        super().__init__(
            f"Image: '{image_path}', with properties: {properties}, is not in part of the supported vendors"
        )

        self._image_path = image_path
        self._properties = properties

    def __reduce__(self):
        return (UnsupportedVendorError, (self._image_path, self._properties))


class SlideReader(object):
    def __init__(self, image_path: str, spacing_tolerance: float = 0.3, cache_path=None, verbose=False) -> None:
        self._image_path = image_path
        self._extension = os.path.splitext(image_path)[-1]
        self._spacing_margin_ratio = spacing_tolerance
        self._cache_path = cache_path
        self._verbose = verbose
        self._cache_image()

    def _init_slide(self):
        self._shapes = self._init_shapes()
        self._downsamplings = self._init_downsamplings()
        self._spacings = self._init_spacings()

    def _get_path(self):
        path = str(self._cache_path) if self._cache_path else str(self._image_path)
        return path

    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self):
        pass

    @property
    def filepath(self) -> str:
        return self._image_path

    @property
    def extension(self) -> str:
        return self._extension

    @property
    def spacings(self) -> List[float]:
        return self._spacings

    @property
    def shapes(self) -> List[Tuple[int, int]]:
        return self._shapes

    @property
    def downsamplings(self) -> List[float]:
        return self._downsamplings

    @property
    def level_count(self) -> int:
        return len(self.spacings)

    def get_downsampling_from_level(self, level: int) -> float:
        return self.downsamplings[level]

    def level(self, spacing: float) -> int:
        spacing_margin = spacing * self._spacing_margin_ratio
        for level, spacing_ in enumerate(self.spacings):
            if abs(spacing_ - spacing) <= spacing_margin:
                return level
        raise InvalidSpacingError(
            self._image_path, spacing, self.spacings, spacing_margin
        )

    def get_downsampling_from_spacing(self, spacing: float) -> float:
        return self.get_downsampling_from_level(self.level(spacing))

    @abstractmethod
    def _init_shapes(self) -> List[Tuple[int, int]]:
        pass

    @abstractmethod
    def _init_downsamplings(self) -> List[float]:
        pass

    @abstractmethod
    def _init_spacings(self) -> List[float]:
        pass

    @abstractmethod
    def get_patch(self, x, y, width, height, spacing, center, relative) -> np.ndarray:
        pass


    def read(self, spacing, r, c, width, height):
        return self.get_patch(c, r, height, width, spacing=spacing, center=False, relative=True)

    def refine(self, spacing):
        """
        Get the pixel spacing of an existing level for the given pixel spacing within tolerance.
        Args:
            spacing (float): Pixel spacing (micrometer).
        Returns:
            float: Best matching pixel spacing of the closest level of the given pixel spacing.
        Raises:
            InvalidSpacingError: There is no level found for the given pixel spacing and tolerance.
        """
        return self.spacings[self.level(spacing=spacing)]

    def _cache_image(self):
        """
        Save the source and cached image paths and copy the image to the cache.

        Args:
            image_path (str): Path of the image to load.
            cache_path (str, None): Directory or file cache path.
        """

        if self._cache_path is None:
            return
        elif str(Path(self._image_path).absolute())==str(Path(self._cache_path).absolute()):
            print('no caching since image and cache are the same:', self._image_path)
            self._cache_path = None
            return

        if not os.path.isfile(self._cache_path):
            # Copy the source image to the cache location.
            cache_target = self._cache_path if os.path.splitext(self._image_path)[1].lower() == os.path.splitext(self._cache_path)[
                1].lower() else os.path.join(self._cache_path, os.path.basename(self._image_path))
            self._cache_path = cache_target
            # cached = copy_image(source_path=self._image_path, target_path=self._cache_path, overwrite=False)
            cached = copy_slide(self._image_path, self._cache_path, overwrite=False)
            if self._verbose and cached:
                print('cached %s to %s' % (self._image_path,self._cache_path), flush=True)

        else:
            if self._verbose:
                print('not caching %s since cache_path %s already exists' %\
                      (Path(self._image_path).name, self._cache_path))

    def close(self, clear=True):
        if clear and self._cache_path is not None:
            if self._verbose: print('removing cached %s' % self._cache_path, flush=True)
            delete_slide(self._cache_path)


class OpenSlideReader(SlideReader):
    def __init__(self, image_path, **kwargs):
        SlideReader.__init__(self, str(image_path), **kwargs)
        self._openslide = OpenSlide(self._get_path())
        self.properties = self._openslide.properties
        # OpenSlide.__init__(self, str(self._get_path()))
        self.channels = 3 #supports only 3 channel rgb
        self._init_slide()

    def get_patch(self, x, y, width, height, spacing, center=False, relative=True) -> np.ndarray:
        """ center: samples with x,y as center """
        downsampling = int(self.get_downsampling_from_spacing(spacing))
        level = self.level(spacing)

        if relative:
            x, y = x * downsampling, y * downsampling
        if center:
            x, y = x - downsampling * (width // 2), y - downsampling * (height // 2)

        return np.array(
            self._openslide.read_region((int(x), int(y)), int(level), (int(width), int(height)))
        )[:, :, :3]

    def _init_shapes(self) -> List[Tuple[int, int]]:
        return self._openslide.level_dimensions

    def _init_downsamplings(self) -> List[float]:
        return self._openslide.level_downsamples

    def _init_spacings(self) -> List[float]:
        spacing = None
        properties = self._openslide.properties
        try:
            spacing = float(properties["openslide.mpp-x"])
        except KeyError as key_error:
            try:
                unit = {"cm": 10000, "centimeter": 10000}[
                    properties["tiff.ResolutionUnit"]
                ]
                res = float(properties["tiff.XResolution"])
                spacing = unit / res
            except KeyError as key_error:
                raise UnsupportedVendorError(
                    self._image_path, properties
                ) from key_error

        spacings = [
            spacing * self.get_downsampling_from_level(level)
            for level in range(self._openslide.level_count)
        ]
        return spacings

    def content(self, spacing=None):
        """
        Load a the content of the complete image from the given pixel spacing.
        Args:
            spacing (float): Pixel spacing to use to find the target level (micrometer).
        Returns:
            (np.ndarray): The loaded image.
        Raises:
            InvalidSpacingError: There is no level found for the given pixel spacing and tolerance.
        """

        if spacing is None:
            spacing = self.spacings[0]

        level = self.level(spacing=spacing)
        shape = self.shapes[level]
        content = self.get_patch(int(0), int(0), int(shape[0]), int(shape[1]), spacing)
        return content

    def close(self, clear=True):
        self._openslide.close()
        # super(OpenSlide, self).close()
        super().close(clear=clear)

    def get_image_label(self, size=128):
        """ non-anonymized mrxs can have an image label"""
        # num = reader._openslide.properties['mirax.NONHIERLAYER_1_LEVEL_3_SECTION.BARCODE_VALUE']
        # props = list(reader.properties)
        img = self._openslide.associated_images.get('label',None)
        if img is None:
            return None
        # img =  Image.fromarray(img_arr)
        res = img.thumbnail((size,size), Image.ANTIALIAS)
        if res is not None:
            img = res #old version?
        img = np.array(img)
        # img.show()
        # reader.close()
        return img

class AsapReader(MultiResolutionImage, SlideReader):
    def __init__(self, image_path: str, **kwargs) -> None:
        SlideReader.__init__(self, image_path, **kwargs)
        self.__dict__.update(MultiResolutionImageReader().open(image_path).__dict__)
        self._init_slide()
        self.setCacheSize(0)

    def get_patch(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        spacing: float,
        center: bool = True,
        relative: bool = False,
    ) -> np.ndarray:

        downsampling = int(self.get_downsampling_from_spacing(spacing))
        level = self.level(spacing)
        if relative:
            x, y = x * downsampling, y * downsampling
        if center:
            x, y = x - downsampling * (width // 2), y - downsampling * (height // 2)

        return np.array(
            super().getUCharPatch(int(x), int(y), int(width), int(height), int(level))
        )

    def _init_shapes(self) -> List[Tuple]:
        try:
            return [
                tuple(self.getLevelDimensions(level))
                for level in range(self.getNumberOfLevels())
            ]
        except:
            raise ValueError("shape en level errors")

    def _init_downsamplings(self) -> List[float]:
        return [
            self.getLevelDownsample(level) for level in range(self.getNumberOfLevels())
        ]

    def _init_spacings(self) -> List[float]:
        try:
            return [
                self.getSpacing()[0] * downsampling
                for downsampling in self.downsamplings
            ]
        except:
            raise InvalidSpacingError(self._image_path, 0, [], 0)

    def close(self, clear=True):
        super(OpenSlide, self).close()
        super(SlideReader, self).close(clear=clear)

def _slide_info(path):
    wsi = OpenSlideReader(str(path))
    print('spacings:', wsi.spacings, 'shapes', wsi.shapes)
    wsi.close()

def slide_infos(dir):
    pathes = PathUtils.list_pathes(dir, containing_or=['.tif','svs'])
    entries = []
    for i,p in enumerate(pathes):
        reader = OpenSlideReader(p)
        spacing = reader.spacings[0]
        w = reader.shapes[0][0]
        h = reader.shapes[0][1]
        entries.append(dict(name=p.stem, spacing=spacing, w=w, h=h))
        print(entries[-1])
        reader.close()

    df = pd.DataFrame(entries)
    print_df(df)


def read_wsi_spacing_os(slide_path):
    reader = OpenSlideReader(slide_path)
    spacing = reader.spacings[0]
    return spacing