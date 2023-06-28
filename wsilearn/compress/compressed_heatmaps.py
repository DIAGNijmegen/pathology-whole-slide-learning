import numpy as np
from pathlib import Path

import scipy
from torch import is_tensor

from wsilearn.dataconf import TrainType
from wsilearn.wsi.wsd_image import ImageReader, PixelSpacingLevelError

from wsilearn.utils.asap_links import make_asap_link, make_link
from wsilearn.utils.cool_utils import ensure_dir_exists, is_iterable
from wsilearn.dl.torch_utils import to_numpy
from wsilearn.wsi.wsi_utils import write_wsi_heatmap, write_salient_patches
from wsilearn.compress.compressed_data import H5Features
from wsilearn.compress.compressed_dataset import CompressedDataset


def write_heatmaps(A, data, out_dir,
                preds=None, class_names=None, anno_dir=None, hmsoft=False, overwrite=False, train_type=TrainType.clf):
    """ writes the heatmaps and salient patches to out_dir
    makes links for wrong predictions if 'label' in data:
        for clf, preds are the predicted labels
        todo: regression: largest diff between preds and target? """


    if out_dir is None: raise ValueError('out_dir is None!')
    ensure_dir_exists(out_dir)

    dkeys = sorted(list(data.keys()))
    wsi_path = CompressedDataset.get_wsi_path(data)
    # wsi_path_col = CompressedDataset.wsi_path_col
    # if CompressedDataset.wsi_path_col not in data:
    #     wsi_path_col = CompressedDataset.wsi_path_col_old
    # wsi_path = data[wsi_path_col]
    coords = data[CompressedDataset.coords_col]
    used_spacing = data['spacing']
    fshape = data['shape']
    stride = data['stride']

    print(f'writing heatmap for {str(wsi_path)}, shape {fshape}, stride {stride}')
    if (is_iterable(stride) and not is_tensor(stride)) or (is_tensor(stride) and len(stride.size())>0):
        coords = coords[0]
        fshape = fshape[0]
        stride = stride[0]
        used_spacing = used_spacing[0]
        if is_iterable(wsi_path): wsi_path = wsi_path[0]
        assert len(coords)>1, print(data)

    fshape = [int(fshape[0]), int(fshape[1])]
    stride = int(stride)
    used_spacing = float(used_spacing)
    if is_tensor(coords):
        coords = to_numpy(coords)

    slide_id = Path(wsi_path).stem
    try:
        reader = ImageReader(str(wsi_path))
    except:
        print('failed opening', wsi_path, 'data keys:', dkeys)
        raise
    # fshape = reader.shapes[reader.level(used_spacing)]
    # if fshape is None:
    #     fshape = np.ceil(np.array(wshape)/stride)
    #     fshape = (int(fshape[0]),int(fshape[1]))
    # if fshape[0]!=fshape2[0]:
    #     print('!!!')
    #     raise
    if len(A.shape) == 1:
        A = A[None, :]

    ensure_dir_exists(out_dir)

    anno_path = None
    if anno_dir is not None and str(anno_dir)!='None' and len(str(anno_dir)) > 1:
        anno_path = Path(anno_dir) / (slide_id + '.xml')
        if not anno_path.exists():
            anno_path = None

    for i in range(A.shape[0]):
        class_suffix = '' if A.shape[0] == 1 else str(i)
        Ai = A[i]
        if is_tensor(Ai):
            Ai = to_numpy(Ai)
        hm_dir_name = ('heatmaps' + class_suffix)
        hm_dir = Path(out_dir) / hm_dir_name
        hm_dir.mkdir(exist_ok=True, parents=True)
        hm_path = hm_dir / (slide_id + '_likelihood_map.tif')
        if overwrite or not hm_path.exists():
            print('creating heatmap for slide %s channel %d' % (slide_id, i))
            hmap = H5Features.stitch_features_from(Ai, coords, downscale=stride, fshape=fshape[:2])
            write_wsi_heatmap(hmap, hm_path, used_spacing, wsi=reader, links_dir=hm_dir, shift=stride,
                              anno_path=anno_path, overwrite=overwrite)

        Ai_soft = scipy.special.softmax(Ai)
        hmap_soft = H5Features.stitch_features_from(Ai_soft, coords, downscale=stride, fshape=fshape[:2])
        hm_dir_soft_name = ('heatmaps_softmax' + class_suffix)
        hm_dir_soft = Path(out_dir) / hm_dir_soft_name
        hm_path_soft = hm_dir_soft / (slide_id + '_likelihood_map.tif')
        if hmsoft:
            hm_dir_soft.mkdir(exist_ok=True, parents=True)
            write_wsi_heatmap(hmap_soft, hm_path_soft, used_spacing, wsi=reader, links_dir=hm_dir_soft,
                              shift=stride, anno_path=anno_path, overwrite=overwrite)
        patch_dir_name = ('patches' + class_suffix)
        patches_dir = Path(out_dir) / patch_dir_name
        pout_path = write_salient_patches(reader, hmap_soft, used_spacing, out_dir=patches_dir, shift=stride,
                                          overwrite=overwrite)
        #make true/wrong combination dir with links
        if train_type in [TrainType.clf,TrainType.multilabel] \
                and preds is not None and CompressedDataset.target_col in data and pout_path is not None:
            gt = data[CompressedDataset.target_col]
            if is_tensor(gt):
                gt = to_numpy(gt)
            gt = np.where(gt[0])[0]
            if not is_iterable(gt):
                gt = [gt]
            if not is_iterable(preds):
                preds = [preds]
            #for multilabel assume large label value to be the more important one
            preds = sorted(preds, reverse=True)
            gt = sorted(gt, reverse=True)
            name = ''
            for gti in gt:
                true_name = class_names[gti]
                name+=true_name+'_'
            for predi in preds:
                pred_name = class_names[predi]
                name+='_'+pred_name
            name = name.replace('__','-')
            links_dir = Path(str(out_dir)+'_links')/name
            hm_links_dir = links_dir/hm_dir_name
            ensure_dir_exists(hm_links_dir)
            make_asap_link(wsi_path, anno_path, hm_path, hm_links_dir, relative='anno', exist_ok=True)
            if hmsoft:
                hm_soft_links_dir = links_dir/hm_dir_soft_name
                ensure_dir_exists(hm_soft_links_dir)
                make_asap_link(wsi_path, anno_path, hm_path_soft, hm_soft_links_dir, relative='anno', exist_ok=True)
            patches_links_dir = links_dir/patch_dir_name
            ensure_dir_exists(patches_links_dir)
            # print('LINKING ', str(pout_path), 'TO', str(patches_links_dir))
            make_link(pout_path, patches_links_dir/pout_path.name, relative=True, exist_ok=True)

    reader.close()