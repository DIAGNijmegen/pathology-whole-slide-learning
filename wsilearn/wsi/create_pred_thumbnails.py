import sys

import traceback
from pathlib import Path
from PIL import Image
import numpy as np
import cv2 as cv
from tqdm import tqdm

from wsilearn.utils.cool_utils import ensure_dir_exists, remap_arr, is_string, is_string_or_path, showims, showim
from wsilearn.utils.parse_utils import parse_string
from wsilearn.wsi.wsi_utils import write_arr_as_image
from wsilearn.utils.flexparse import FlexArgumentParser
from wsilearn.utils.path_utils import get_corresponding_pathes_dirs, PathUtils, get_path_named_like
from wsilearn.wsi.wsd_image import ImageReader

asap_colors = [(0, 224, 249),(0, 249, 50), (174, 249, 0),(249, 100, 0),(249, 0, 125),(149, 0, 249),
               (0, 0, 207), (0, 185, 206), (0, 207, 41), (143, 206, 3), (207, 82, 0), (206, 0, 103)]


def _write_image(arr, path):
    if arr.shape[-1]==1:
        arr = arr.squeeze()
    if 'uint' in str(arr.dtype):
        arr*=(255//arr.max())
    else:
        print('dtype of mask not uint, but %s!' % str(arr.dtype))
    #mask
    im = Image.fromarray(arr).convert('RGB')
    args = {}
    if str(path).lower().endswith('jpeg') or str(path).lower().endswith('jpg'):
        args = dict(quality=80)
    im.save(str(path), **args)
    print('%s saved' % Path(path).absolute())


def create_pred_thumbnail(mask_path, out_dir, wsi_path=None, alpha=0.3, overwrite=False, duo=False, level=-1, spacing=None,
                          remap=None, heatmap=False, name=None, spacing_tolerance=0.3, **kwargs):
    if name is None:
        name = Path(mask_path).stem if (wsi_path is None or not is_string_or_path(wsi_path)) else Path(wsi_path).stem
    out_path = Path(out_dir)/(name+'.jpg')
    if out_path.exists() and not overwrite:
        print('skipping existing %s' % out_path)
        return out_path, False

    ensure_dir_exists(out_dir)

    if spacing is None and wsi_path is not None:
        if is_string_or_path(wsi_path):
            wsi_reader = ImageReader(str(wsi_path), spacing_tolerance=spacing_tolerance)
            spacing = wsi_reader.spacings[level]
            wsi_reader.close()
        else:
            spacing = wsi_path.spacings[level]
    # print('crating pred thumbnail for %s' % mask_path)
    reader = ImageReader(str(mask_path), spacing_tolerance=spacing_tolerance)
    if spacing is None:
        spacing = reader.spacings[level]
    mask = reader.content(spacing).squeeze()
    reader.close()

    return _create_pred_thumbnail(mask, spacing, out_dir=out_dir, wsi_path=wsi_path, alpha=alpha, overwrite=overwrite, duo=duo, level=level,
                           remap=remap, heatmap=heatmap, name=name, **kwargs)

def _create_pred_thumbnail(mask, spacing, out_dir, name, wsi_path=None, alpha=0.3, overwrite=False, duo=False, level=-1,
                           remap=None, heatmap=False, heatmap_cmap='jet'):
    out_path = Path(out_dir)/(name+'.jpg')
    if out_path.exists() and not overwrite:
        print('skipping existing %s' % out_path)
        return out_path, False
    if remap is not None and is_string(remap):
        remap = parse_string(remap)
    remap_arr(mask, remap)

    if heatmap:
        segm = np.uint8(255 * mask) if mask.dtype in [np.float, float] else mask
        segm_bgr = cv.cvtColor(segm, cv.COLOR_RGB2BGR)
        if heatmap_cmap.lower()=='reds':
            import cmapy
            segm_bgr = cv.applyColorMap(segm_bgr, cmapy.cmap('Reds'))
        elif heatmap_cmap.lower()=='jet':
            segm_bgr = cv.applyColorMap(segm_bgr, cv.COLORMAP_JET)
        else: raise ValueError('only reds and jet supported')
        segm =  cv.cvtColor(segm_bgr, cv.COLOR_BGR2RGB)
    else:
        segm = np.ones((mask.shape[0],mask.shape[1],3), dtype=np.uint8)*255
        for i in range(1,mask.max()+1):
            color = np.array(asap_colors[i-1])
            segm[mask==i] = color

    if wsi_path is not None:
        if is_string_or_path(wsi_path):
            reader = ImageReader(str(wsi_path))
            close = True
        else:
            reader = wsi_path
            close = False
        try:
            slide = reader.content(spacing)
            if close: reader.close()
            overlay=cv.addWeighted(segm, alpha, slide, 1.0-alpha, 0)
            # segm[mask==i] = overlay[mask==i]
            segm = overlay
        except:
            print('getting content from slide %s at spacing %f failed, slide spacings: %s' %\
                  (str(wsi_path), spacing, str(reader.spacings)))
        finally:
            reader.close()


    if duo:
        #it can be that the segm is slighlty larger due to rounding up to patch_size
        if slide.shape[0]!=segm.shape[0]:
            segm = segm[:slide.shape[0]]
        if slide.shape[1]!=segm.shape[1]:
            segm = segm[:,:slide.shape[1]]
        segm = np.concatenate((slide, segm), axis=1)
    write_arr_as_image(segm, out_path)
    return out_path, True

def create_pred_thumbnails(mask_dir, out_dir=None, wsi_dir=None, alpha=0.5, duo=False, overwrite=False,
                           level=-1, must_all_match=False, **kwargs):
    if wsi_dir is not None:
        wsi_pathes, mask_pathes = get_corresponding_pathes_dirs(wsi_dir, mask_dir, take_shortest=True,
                                    not_containing1='likelihood_map,xml,json', must_all_match=must_all_match,
                                    ignore_missing=True, ignore_missing2=False, ending2='tif')
    else:
        mask_pathes = PathUtils.list_pathes(mask_dir, ending='tif')
    print('creating thumbnails for %d files in %s' % (len(mask_pathes), mask_dir))
    if out_dir is None:
        out_dir = str(Path(mask_dir))+'_thumbnails'
        if duo:
            out_dir+='_duo'
    print('out_dir:', out_dir)

    ensure_dir_exists(out_dir)
    failures = []; skipped = []
    counter=0
    for i,mask_path in tqdm(enumerate(mask_pathes)):
        mask_name = mask_path.stem
        # wsi_path = get_path_named_like(mask_name.replace('_pred',''), wsi_pathes, same_name=True)
        esi_path = None
        if wsi_dir is not None:
            wsi_path = wsi_pathes[i]
        try:
            thumb_path, thumb_created = create_pred_thumbnail(mask_path, out_dir=out_dir, overwrite=overwrite, wsi_path=wsi_path,
                              alpha=alpha, duo=duo, level=level, **kwargs)
            if thumb_created:
                counter+=1
        except:
            print('FAILED %s' % mask_name)
            failures.append(mask_path)
            traceback.print_exc()

    if len(failures)>0:
        print('%d failures' % len(failures))

    for f in failures:
        print('rm %s' % str(f))
    print('Done! %d thumbnails created' % counter)



if __name__ == '__main__':
    print('sys.argv:', str(sys.argv))
    parser = FlexArgumentParser()
    parser.add_argument('--mask_dir', required=True, type=str)
    parser.add_argument('--wsi_dir', required=False, type=str, default=None)
    parser.add_argument('--out_dir', required=False, default=None, type=str)
    parser.add_argument('--remap', required=False, type=str, help='e.g. {255:1}')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    create_pred_thumbnails(**args)

