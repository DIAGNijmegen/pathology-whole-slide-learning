from copy import deepcopy

from albumentations import HorizontalFlip, VerticalFlip, Rotate, Compose, ShiftScaleRotate, HueSaturationValue, \
    RandomBrightnessContrast, GaussianBlur, JpegCompression, ImageCompression

from wsilearn.utils.cool_utils import print_mp
from wsilearn.dataconf import DataConf
from wsilearn.utils.flexparse import FlexArgumentParser

print = print_mp

print('entered compress.py')

import traceback
from argparse import ArgumentParser

import torch
import torch.nn.parallel
import pandas as pd

from wsilearn.utils.cool_utils import *
# print_env_info()

from wsilearn.wsi.wsd_image import ImageReader, PixelSpacingLevelError

from wsilearn.utils.path_utils import *
from wsilearn.dl.torch_utils import determine_max_input_volume
from wsilearn.utils.gpu_utils import gpu_mem
from wsilearn.utils.signal_utils import ExitHandler
from wsilearn.wsi.wsi_utils import read_patch_from_arr
from wsilearn.wsi.wsi_read import OpenSlideReader
from wsilearn.compress.compressed_data import compressed_slide_from_ending, CompressedInfo
from wsilearn.compress.encoders_create import create_encoder


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

patch_encoding_map = {'none':AugmentedEncoding, 'rot8m':RotationMeanEncoding, 'color8m':ColorMeanEncoding,
                      'raug1':RandomAugmentationEncoding1, 'rc4m':RotationColorMeanEncoding}

class SlideCompressor(object):
    #former Featurizer

    #columsn in the result df
    ok_col = 'ok'
    existed_col = 'existed'
    locked_col = 'locked'
    error_col = 'error'
    out_path_col = 'out_path'

    def __init__(self, batch_size, out_dir, input_size, shift, out_format='h5',
                 spacing=0.5, spacing_tolerance=0.3, mask_spacing=2.0, mask_label=None, cache_dir=None,
                 augm=None,
                 overwrite=False, work_dir=None,
                 mask_thresh=0.2, #mask_thresh=0.6,
                 reader='asap',
                 fp32=False, pred_center_crop=None,
                 suppress_input_name = False,
                 thumbs=False, thumbs_idx=None,
                 ):
        self.batch_size = batch_size
        self.out_dir = Path(out_dir)
        self.patch_size = input_size
        self.shift = shift
        self.augm_encoding_cls = patch_encoding_map[str(augm).lower()]
        self.out_format = out_format
        self.spacing = spacing
        self.spacing_tolerance = spacing_tolerance
        if str(mask_spacing).lower()=='none':
            mask_spacing = None
        self.mask_spacing_init = mask_spacing
        self.cache_dir = cache_dir
        # self.normalizer = create_input_normalizer(normalizer)
        self.overwrite = overwrite
        self.work_dir = work_dir
        self.mask_thresh = mask_thresh
        self.mask_label = mask_label
        self.pred_center_crop = pred_center_crop
        self.suppress_input_name = suppress_input_name

        self.thumbs = thumbs
        self.thumbs_idx = thumbs_idx

        self._encoder_time = 0
        self._mask_read_time = 0
        self._slide_read_time = 0
        self._overall_time = 0

        if reader=='asap':
            self._reader_class = ImageReader
        elif reader=='openslide':
            self._reader_class = OpenSlideReader
        else:
            raise ValueError('reader %s not asap or openslide' % reader)

        # self.read_whole_slide = read_whole_slide
        # self.read_whole_mask = read_whole_mask

        self.save_float = 'float16'
        if fp32:
            self.save_float = 'float32'
        print(f'Compressor batch_size {batch_size}, patch_size {self.patch_size} stride {shift}'
              # f' normalizer {normalizer}'
              f', mask_thresh {mask_thresh}, fp {self.save_float}')
        print('out dir:', out_dir)
        if shift > input_size:
            raise ValueError('expects shift %d to be smaller or equal to input size %d' % (shift, input_size))

        # invert_op = getattr(encoder, "print_model_summary", None)
        # if callable(invert_op):
        #     encoder.print_model_summary([input_size, input_size, 3])

        # if channels_first:
        #     input_patch_shape = (len(input_channels), patch_size, patch_size)
        # else:
        #     input_patch_shape = (patch_size, patch_size, len(input_channels))
        # network_model.print_model_summary(input_patch_shape)

    def _print_time(self):
        print('Time overall: %s, read slide: %s, read mask: %s, encode: %s' % \
              (time_to_str(self._overall_time), time_to_str(self._slide_read_time), \
               time_to_str(self._mask_read_time), time_to_str(self._encoder_time)))

    def compress_patches(self, encoder, patches):
        if len(patches) == 1:
            batch = np.expand_dims(patches[0], 0)
        elif len(patches) > 1:
            batch = np.array(patches)
        else:
            return None

        # if self.normalizer is not None:
        #     if batch.dtype != np.uint8: raise ValueError('unknown input type %s' % str(batch.dtype))
        #     batch = self.normalizer(batch)

        start_time = time.time()
        augm_encoding = self.augm_encoding_cls(encoder)
        # features = encoder(batch)
        features = augm_encoding(batch)

        self._encoder_time += (time.time()-start_time)
        return features

    def read_mask_patch(self, mask, row_ind, col_ind, coord_spacing, mask_spacing=None, mask_content=None):
        """ row and col are at coord_spacing"""
        if mask is None: return None
        if mask_spacing is None:
            mask_spacing = self.mask_spacing_init
        if mask_spacing is None:
            mask_spacing = take_closest_larger_number(mask.spacings, 3)
        if mask_spacing is None:
            mask_spacing = mask.spacings[0]
        mask_spacing = mask.refine(mask_spacing)
        downsampling = coord_spacing/mask_spacing
        m_row_ind = int(np.round(row_ind*downsampling))
        m_col_ind = int(np.round(col_ind*downsampling))
        m_shift = int(np.round(self.patch_size*downsampling))
        try:
            if mask_content is None:
                mpatch = mask.read(mask_spacing, m_row_ind, m_col_ind, m_shift, m_shift)
            else:
                mpatch = read_patch_from_arr(mask_content, m_row_ind, m_col_ind, m_shift, m_shift)
        except:
            print(f'failed reading f mask {mask.path} at spacing {mask_spacing}, {m_row_ind}, {m_col_ind}, {m_shift}, {m_shift}')
            print('mask shapes:', mask.shapes, 'mask spacings:', mask.spacings)
            raise
        return mpatch

    def _create_out_path(self, slide_path):
        result_name = Path(slide_path).stem
        if self.suppress_input_name:
            result_name = ''

        if self.suppress_input_name and result_name.startswith('_'):
            result_name = result_name[1:]
        out_path = self.out_dir / (result_name + '.'+self.out_format).replace('..','.')
        return out_path

    def featurize_slide(self, encoder, slide_path, mask_path, out_path=None):
        print('compress slide %s' % (slide_path))

        ensure_dir_exists(self.out_dir)
        if out_path is None:
            out_path = self._create_out_path(slide_path)

        result_info = dict(slide_path=str(slide_path), mask_path=str(mask_path), error=None,
                           locked=False, existed=False, ok=True)
        result_info['out_path'] = str(out_path)

        lock_path = Path(str(out_path)+'.lock')
        if out_path.exists() and not self.overwrite:
            print('skipping existing %s' % out_path)
            result_info[self.existed_col] = True
        elif can_open_file(lock_path):
            print('skipping locked %s' % lock_path)
            result_info[self.locked_col] = True
            result_info[self.ok_col] = False
        else:
            lock_path.touch()
            path_unlinker = ExitHandler.instance().add_path_unlinker(lock_path)
            try:
                fshape = self._featurize_slide(encoder, slide_path, mask_path, out_path)
                result_info['shape'] = fshape
            except Exception as ex:
                print('Failed %s with exc %s' % (str(slide_path), str(ex)))
                traceback.print_exc(file=sys.stdout)
                result_info[self.ok_col] = False
                result_info[self.error_col] = str(ex)
                # print('DEBUG!')
                # raise
                return result_info
            finally:
                try:
                    lock_path.unlink()
                    ExitHandler.instance().remove(path_unlinker)
                except Exception as ex:
                    print('Failed to remove %s' % str(lock_path))
                    print(sys.exc_info())
        print('finished %s' % slide_path)
        return result_info


    def _featurize_slide(self, encoder, slide_path, mask_path, out_path):
        if out_path.exists() and not self.overwrite:
            raise ValueError('%s already exists!' % str(out_path))

        overall_start = time.time()
        print('out: %s' % (out_path))
        diff_half = (self.patch_size - self.shift) // 2
        if self.cache_dir is not None and len(str(self.cache_dir))>0:
            ensure_dir_exists(self.cache_dir)

        slide = self._reader_class(str(slide_path), cache_path=self.cache_dir, verbose=True, spacing_tolerance=self.spacing_tolerance)
        if mask_path is not None and len(str(mask_path))>1:
            mask = ImageReader(str(mask_path), cache_path=self.cache_dir, spacing_tolerance=self.spacing_tolerance)
        else:
            mask = None
            print('No mask given for %s' % str(slide_path))
        example = np.random.randn(2, self.patch_size, self.patch_size, slide.channels).astype(np.float32)
        example_encoding = encoder(example)
        if len(example_encoding.shape)!=2:
            raise ValueError('unknown example_encoding shape %s for input shape %s' \
                             % (str(example_encoding.shape),str(example.shape)))
        code_size = example_encoding.shape[1]

        slide_level = slide.level(self.spacing)
        spacing = slide.refine(self.spacing)
        slide_shape = slide.shapes[slide_level]  # rows,cols (height, width)
        print('slide spacing %.1f: shape %s, level %d' % (self.spacing, slide_shape, slide_level))

        # mask_spacing = mask.spacings[0]
        # mask_shape = mask.shapes[0]

        # print('slide shapes', slide.shapes)
        # print('mask spacing %.1f: shape %s' % (mask_spacing, mask_shape))


        slide_inner_row = np.array(range(0, slide_shape[0], self.shift))
        slide_inner_col = np.array(range(0, slide_shape[1], self.shift))

        rows_arr, cols_arr = np.meshgrid(slide_inner_row, slide_inner_col, indexing='ij')
        # rows_arr = rows_arr_inner-diff_half
        # cols_arr = cols_arr_inner - diff_half

        out_shape = np.ceil(np.array(slide_shape) / self.shift).astype(np.uint8)
        print('featurize slide %s with shape %s, out_shape %s, ps=%d, stride=%d, batch_size=%d' %\
              (Path(slide_path).name, str(slide_shape), str(out_shape), self.patch_size, self.shift, self.batch_size))

        result_size = rows_arr.shape[0], rows_arr.shape[1]
        result = np.zeros((result_size[0], result_size[1], code_size), dtype=example_encoding.dtype)
        tissue = np.zeros((result_size[0], result_size[1]), dtype=np.uint8)

        idxs_flat_arr = np.arange(result_size[0] * result_size[1]).reshape((result_size[0], result_size[1]))

        self._encoder_time = 0
        self._mask_read_time = 0
        self._slide_read_time = 0

        print('featurize...')
        batch = []; batch_inds = []
        mask_spacing = self.mask_spacing_init
        for i in range(len(rows_arr)):
            if i>0 and i%20==0:
                print('%d/%d rows written' % (i, len(rows_arr)))
            for j in range(rows_arr.shape[1]):
                start = time.time()
                if mask is None:
                    mpatch = None
                else:
                    for z in range(2):
                        try:
                            # mpatch = self.read_mask_patch(mask, rows_arr[i,j], cols_arr[i,j], coord_spacing=slide_spacing)
                            mpatch = self.read_mask_patch(mask, rows_arr[i,j], cols_arr[i,j],
                                                          coord_spacing=spacing, mask_spacing=mask_spacing)
                            break
                        except PixelSpacingLevelError as e:
                            print(f'PixelSpacingLevelError when trying to read mask at spacing {mask_spacing:.1f}')
                            if mask_spacing > (self.mask_spacing_init*1.9):
                                print(f'spacing already increased from initially {spacing}')
                                raise e
                            else:
                                mask_spacing = self.mask_spacing_init * 2
                                print(f'increasing mask_spacing to {mask_spacing:.1f}')
                                #now try again

                    self._mask_read_time += (time.time() - start)
                # if (mpatch>0).any():
                if mpatch is not None and self.mask_label is not None:
                    mpatch = mpatch==self.mask_label

                if mpatch is None or np.mean(mpatch>0) > self.mask_thresh:
                    r = rows_arr[i, j] - diff_half
                    c = cols_arr[i, j] - diff_half
                    start = time.time()
                    # if slide_content is None:
                    patch = slide.read(spacing, r, c, self.patch_size, self.patch_size)
                    # if slide_content is not None:
                        # patch = read_patch_from_arr(slide_content, r, c, self.patch_size, self.patch_size)
                    self._slide_read_time += (time.time() - start)
                    batch.append(patch)
                    batch_inds.append((i,j))
                    tissue[i,j] = 1

                if len(batch)==self.batch_size or \
                        (i==len(rows_arr)-1 and j==rows_arr.shape[1]-1 and len(batch)>0):

                    features = self.compress_patches(encoder, batch)

                    for b, (bi, bj) in enumerate(batch_inds):
                        #now assign the (rotated and flipped) features to the correct spot in features array
                        flat_ind = bi*result_size[1]+bj
                        rot_coords = np.where(idxs_flat_arr==flat_ind)
                        rot_bi, rot_bj = rot_coords[0][0], rot_coords[1][0]
                        result[rot_bi, rot_bj] = features[b]
                    batch = []; batch_inds = []
        slide.close()
        if mask is not None:
            mask.close()
            n_patches = np.sum(tissue)
            print('processed %d patches' % n_patches)
            if n_patches==0:
                print('WARNING: NO PATCHES in %s' % str(slide_path))

        ensure_dir_exists(self.out_dir)
        if str(result.dtype)!=self.save_float:
            result = result.astype(self.save_float)
        formatter = compressed_slide_from_ending(self.out_format, path=out_path, cache_dir=self.cache_dir,
                                                 overwrite=self.overwrite)
        formatter.write(result, spacing=spacing, patch_size=self.patch_size, stride=self.shift)

        if self.thumbs:
            thumbs_out = Path(str(self.out_dir)+'_thumbs')
            formatter.show(show=False, idx=self.thumbs_idx, save_dir=thumbs_out, data=result)

        self._overall_time = (time.time() - overall_start)
        self._print_time()
        # return dict(fshape=result.shape)
        return result.shape



class SlidesCompressor(object):
    def __init__(self, encoder, take_shortest_name=True, only_matching_masks=False, allow_no_masks=False,
                 **featurizer_kwargs):
        self.encoder = encoder
        self.take_shortest = take_shortest_name
        self.featurizer_kwargs = featurizer_kwargs
        self.only_matching_masks = only_matching_masks
        self.allow_no_masks = allow_no_masks

    def _compress_slides_masks(self, slides_masks):
        """ return a df with the result infos"""
        compressor = SlideCompressor(**self.featurizer_kwargs)
        run_est = RunEst(n_tasks=len(slides_masks))
        infos = []
        for i, (slide_path, mask_path) in enumerate(slides_masks):
            print(f'{i + 1}/{len(slides_masks)}')
            run_est.start()
            info = compressor.featurize_slide(self.encoder, slide_path, mask_path)
            run_est.stop(print_remaining_string=True)
            infos.append(info)
        self.df = pd.DataFrame(infos)

    def compress_slide(self, slide, mask):
        self._compress_slides_masks([(slide,mask)])
        return self._check_result_infos()

    def compress_config(self, config_path, purpose=None, mask_dir=None, mask_suffix='_tissue.tif'):
        dfc = pd.read_csv(config_path)
        if purpose is not None:
            dfc = dfc[dfc[DataConf.split_col]==purpose]
        slides = dfc[DataConf.image_col]
        if DataConf.mask_col not in dfc:
            if mask_dir is None or str(mask_dir).lower()=='none':
                raise ValueError('mask_dir is none!')
            masks = [str(Path(mask_dir)/Path(slide).stem+mask_suffix) for slide in slides]
        else:
            masks = dfc[DataConf.mask_col]
        slides_masks = list(zip(slides, masks))
        with timer('compressing'):
            self._compress_slides_masks(slides_masks)
        ok = self._check_result_infos()
        return ok

    def compress_dir(self, slides_dir, masks_dir, config=None):
        containing_or = ['.tif', '.svs', '.mrxs']
        if self.only_matching_masks:
            slide_pathes, mask_pathes = get_corresponding_pathes_dirs(slides_dir, masks_dir,
                                                    containing1=containing_or, ending2='tif')
        else:
            slide_pathes = PathUtils.list_pathes(slides_dir, containing_or=containing_or, sort=True)
            masks_pathes = PathUtils.list_pathes(masks_dir, ending='.tif')
            if self.allow_no_masks:
                slide_pathes, mask_pathes = get_corresponding_pathes(slide_pathes, masks_pathes, must_all_match=False, take_shortest=self.take_shortest)
            else:
                mask_pathes = get_corresponding_pathes_all(slide_pathes, masks_pathes, take_shortest=self.take_shortest)
        print('featurizing %d pathes' % len(slide_pathes))

        # run_est = RunEst(n_tasks=len(slide_pathes))
        slides_masks = list(zip(slide_pathes, mask_pathes))
        if config is not None:
            dfc = pd.read_csv(config)
            names = dfc.name.values
            slides_masks = [(slide, mask) for slide,mask in slides_masks if Path(slide).stem in names]
            print('reduced found %d to at most %d' % (len(slide_pathes), len(slides_masks)))
        with timer('compressing'):
            self._compress_slides_masks(slides_masks)
        ok = self._check_result_infos()
        return ok

    def _check_result_infos(self, df=None):
        if df is None: df = self.df
        compressor = SlideCompressor
        print('%d slides, %d ok, %d locked, %d errors' %\
              (len(df), df[compressor.ok_col].sum(), df[compressor.locked_col].sum(),
               df.count()[compressor.error_col]))
        ok = len(df)==0 or len(df)==df[compressor.ok_col].sum()

        if ok:
            ci = CompressedInfo(self.featurizer_kwargs['out_dir'])
            if not (self.df[compressor.existed_col].all() and ci.infos_exist()):
                infos = []
                for idx, row in tqdm(self.df.iterrows(), miniters=5):
                    try:
                        formatter = compressed_slide_from_ending(self.featurizer_kwargs['out_format'],
                                                                 path=row['out_path'])
                        infos.append(formatter.info())
                    except:
                        print('failed getting infos about %s' % str(dict(row)))
                        raise
                df_infos = pd.DataFrame(infos)
                ci.write(df_infos)
        return ok


def compress(data, out_dir, encoder, encoder_path=None, mask_dir=None, config=None, out_format='h5',
             spacing=0.5, patch_size=256, stride=256, batch_size=64, overwrite=False, cache_dir=None,
             augm=None, mask_label=None, mask_spacing=2.0,
             layer_name=None, multiproc=False, fp32=False, purpose=None,
             pred_center_crop=None, take_shortest_name=True, only_matching_masks=False,
             n_cpus=None, n_gpus=None, reserve=0.15, thumbs=True, thumbs_idx=range(64),
             allow_no_masks=False, suppress_input_name=False, clear_locks=False, spacing_tolerance=0.3,
             disable_cudnn=False, **enc_kwargs):
    print('starting compress with encoder %s, spacing=%.2f' % (encoder, spacing))

    if disable_cudnn:
        torch.backends.cudnn.enabled = False
        print('disable cudnn')

    if batch_size=='auto':
        start_size = [1024, 3, patch_size, patch_size]
        model = create_encoder(encoder=encoder, layer_name=layer_name,
                               encoder_path=encoder_path, **enc_kwargs)._create_model()
        if n_gpus is None:
            n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            print('ddp model with %d devices' % n_gpus)
            model = torch.nn.parallel.DataParallel(model)
        else:
            print('determining max input volume with %d gpus' % n_gpus)
        working_size, _ = determine_max_input_volume(model, start_size, train=False, even_batch_size=True,
                                                     reserve=reserve)
        del model
        model = None
        torch.cuda.empty_cache()
        batch_size = working_size[0]
        print('auto-setting batch_size to %d' % working_size[0])
        gpu_mem(print_info=True)
        print("CUDA memory allocated:", torch.cuda.memory_allocated())

    if not disable_cudnn:
        torch.backends.cudnn.benchmark = True

    if stride is None:
        stride = patch_size
    if stride > patch_size:
        raise ValueError('stride %d > patch size %d' % (stride, patch_size))
    elif stride!='patch_size':
        print('stride %d, patch_size %d' % (stride, patch_size))
    # else:
    #     batch_size = adapt_batch_size(batch_size)

    if clear_locks:
        print('clearing locks from %s' % out_dir)
        run_cmd_live(f'find {str(out_dir)}/*.lock -delete')

    encoder_name = str(encoder)
    print('creating encoder %s with kwargs %s' % (str(encoder), str(enc_kwargs)))
    encoder = create_encoder(encoder=encoder, layer_name=layer_name,
                             encoder_path=encoder_path, n_cpus=n_cpus, n_gpus=n_gpus, **enc_kwargs)

    compress_kwargs = dict(batch_size=batch_size, out_dir=out_dir, shift=stride, fp32=fp32,
                           input_size=patch_size, overwrite=overwrite, take_shortest_name=take_shortest_name,
                           out_format=out_format, spacing=spacing, cache_dir=cache_dir, mask_label=mask_label,
                           only_matching_masks=only_matching_masks, allow_no_masks=allow_no_masks, mask_spacing=mask_spacing,
                           suppress_input_name=suppress_input_name, thumbs=thumbs, augm=augm, spacing_tolerance=spacing_tolerance)
    # all_args = dict(encoder=encoder_name, layer_name=layer_name,
    #                 encoder_path=encoder_path, n_cpus=n_cpus, n_gpus=n_gpus, **enc_kwargs)
    # all_args['compress'] = compress_kwargs
    # out_args_path = Path(out_dir)/'compress_args.json'
    # write_json_dict(out_args_path, all_args)

    seqf = SlidesCompressor(encoder, **compress_kwargs, thumbs_idx=thumbs_idx)

    for i in range(2):
        if Path(data).suffix in ['.tif', '.mrxs', '.svs', '.ndpi']:
            ok = seqf.compress_slide(data, mask_dir)
        elif data.endswith('.csv'):
            ok = seqf.compress_config(data, purpose, mask_dir=mask_dir)
        else:
            ok = seqf.compress_dir(data, mask_dir, config=config)

        if ok:
            break
        else:
            print('compression didnt work, repeat')

    return ok


def process_arguments():
    parser = FlexArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='data config csv or directory with wsis')
    parser.add_argument('--mask_dir', type=str, required=False, help='required if data is a directory with wsis')
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--encoder', type=str, required=True,
                        choices=['histossl','mtdp_res50', 'res50', 'res50last','incres2', 'densenet']
                        )
    parser.add_argument('--encoder_path', type=str, required=False)
    parser.add_argument('--cache_dir', type=str, required=False)
    # parser.add_argument('--normalizer', type=str, required=False, help='either 01 or imagenet')
    #set dependent on encoder

    parser.add_argument('--purpose', type=str, required=False, help='only if data is a config')
    parser.add_argument('--out_format', type=str, required=False, default='h5')
    parser.add_argument('--augm', type=str, required=False, default=None)
    parser.add_argument('--spacing', type=float, required=False, default=0.5)
    parser.add_argument('--patch_size', type=int, required=False, default=256)
    parser.add_argument('--n_cpus', type=int, required=False)
    parser.add_argument('--n_gpus', type=int, required=False)
    parser.add_argument('--batch_size', type=str, required=False, default='32',
                        help='"auto" will try to determine the best batch size for the gpu')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--multiproc', action='store_true')

    args = parser.parse_args()
    args['batch_size'] = args.get('batch_size','32')
    if is_string(args['batch_size']) and args['batch_size'].isdecimal():
        args['batch_size'] = int(args['batch_size'])

    print('compress args:')
    print(args)
    out_dir = args.get('out_dir')
    write_json_dict(path=Path(out_dir)/'compress_args.json', data=args)
    compress(**args)


def main():
    process_arguments()

if __name__ == '__main__':
    main()
