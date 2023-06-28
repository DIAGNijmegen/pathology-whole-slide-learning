from collections import defaultdict

import numpy as np
import scipy
from pathlib import Path

from scipy.special import expit

from wsilearn.utils.asap_links import make_asap_link, make_link
from wsilearn.utils.cool_utils import Timer, is_iterable, mkdir, showim
from wsilearn.dataconf import DataConf, TrainType
from wsilearn.dl.models.densenet import DenseNet
from wsilearn.wsi.create_pred_thumbnails import create_pred_thumbnail
from wsilearn.wsi.wsd_image import ImageReader
from wsilearn.wsi.wsi_utils import write_wsi_heatmap, write_salient_patches
from wsilearn.compress.compressed_dataset import CompressedDataset
from wsilearn.att_net import AttNet

def create_nic_model(name, **kwargs):
    if name.lower()=='attnet':
        net = AttNet(**kwargs)
    elif name.lower()=='dense':
        net = DenseNet(**kwargs)
    else:
        raise ValueError('unknown network %s' % name)

    return net

class HeatmapWriter(object):
    def __init__(self, npz_dir, name_col='name', att_keys=None, n_hm_correct_max=50, n_hm_wrong_max=50,
                 out_hm_dir_name='heatmaps', out_patches_dir_name='patches', anno_dir=None, hmraw=False,
                 overwrite=False, no_links=False, train_type=TrainType.clf):
        self.npz_dir = npz_dir
        if att_keys is None:
            att_keys={'A':'scale'}
            if train_type == TrainType.clf:
                att_keys['A_out'] = 'softmax'
            elif train_type == TrainType.multilabel:
                att_keys['A_out'] = 'sigm'
            else:
                att_keys['A_out'] = None
        self.att_keys = att_keys
        for att_op in att_keys.values():
            if att_op not in [None,'None','scale','softmax','sigm']:
                raise ValueError('unknown att operation %s' % str(att_op))

        self.name_col = name_col
        self.n_hm_correct_max = n_hm_correct_max
        self.n_hm_wrong_max = n_hm_wrong_max

        #for links
        self.anno_dir = anno_dir

        self.out_hm_dir_name = out_hm_dir_name
        self.out_patches_dir_name = out_patches_dir_name

        #name->[pathes]
        self.history_hm_pathes = defaultdict(list) #to avoid re-saving heatmaps for overlapping categories
        self.history_hm_thumb_pathes = defaultdict(list) #to avoid re-saving heatmaps thumbs for overlapping categories
        self.history_patch_pathes = defaultdict(list) #to avoid re-saving heatmaps for overlapping categories

        self.overwrite = overwrite
        self.visu_hmscaled = not hmraw
        self.no_links = no_links
        self.disabled = False

        self.timer = Timer('HeatmapWriter', start=False)

    def evaluate(self, data_conf:DataConf, out_dir):
        if self.disabled:
            return
        self.timer.start()
        self._evaluate(data_conf, out_dir)
        self.timer.stop()

    def _evaluate(self, data_conf:DataConf, out_dir):
        #will be run per category, save heatmaps, patches and links

        if out_dir is None:
            return

        counter = 0
        for idx, row in data_conf.df.iterrows():
            #TODO handle the case when there are no targets (save all)
            save = True
            if (counter>self.n_hm_correct_max):
                save = False

            if save:
                self.save_hm_and_patches(row, out_dir, target_cols=data_conf.target_cols)
                self.save_hm_and_patches_links(Path(out_dir) / 'hmp_links', row,
                                               target_cols=data_conf.target_cols)

    def print_time(self):
        return self.timer.print()

    def get_hm_pathes(self):
        return list(self.history_hm_pathes.items())

    def save_hm_and_patches(self, row, out_dir, target_cols=None):
        name = row[self.name_col]

        if name in self.history_hm_pathes:
            hm_pathes = self.history_hm_pathes[name]
            patch_pathes = self.history_hm_pathes[name]
            return hm_pathes, patch_pathes

        wsi_path = CompressedDataset.get_wsi_path(row)
        # coords = data[CompressedDataset.coords_col]
        used_spacing = row['spacing']
        # fshape = data['shape']
        stride = row['stride']
        patch_size = row['patch_size']

        self.save_hm_and_patches_with(name=name, wsi_path=wsi_path, used_spacing=used_spacing,
                                      out_dir=out_dir, stride=stride, patch_size=patch_size, target_cols=target_cols)

    def save_hm_and_patches_with(self, name, out_dir, wsi_path=None, used_spacing=None, stride=None, patch_size=None, target_cols=None):
        if self.npz_dir is None:
            raise ValueError('npz_dir is None')
        else:
            self.npz_dir = Path(self.npz_dir)
        path = self.npz_dir/(str(name)+'.npz')
        file = np.load(str(path))
        if wsi_path is None:
            wsi_path = file['path']
        if used_spacing is None:
            used_spacing = file['spacing']
        if stride is None:
            stride = file['stride']
        if patch_size is None:
            patch_size = file['patch_size']

        reader = ImageReader(str(wsi_path))
        anno_path = self._get_anno_path(name)

        for akey,operation in self.att_keys.items():
            if akey not in file: continue
            A = file[akey]

            if operation=='softmax' and A.shape[0]>1:
                A = scipy.special.softmax(A, axis=0)

            midfix = akey.replace('A','')
            for i in range(A.shape[0]):
                if A.shape[0]==1:
                    class_suffix = ''
                else:
                    if target_cols is None:
                        class_suffix = '_%d' % i
                    else:
                        class_suffix = '_%s' % target_cols[i]

                hm_dir_name = ('heatmaps' + midfix + class_suffix)
                hm_dir = Path(out_dir) / hm_dir_name
                hm_dir.mkdir(exist_ok=True, parents=True)

                print('creating heatmap for slide %s channel %d' % (name, i))

                # hmap = H5Features.stitch_features_from(Ai, coords, downscale=stride, fshape=fshape[:2])
                Ai = A[i]

                patch_dir_name = ('patches' + midfix + class_suffix)
                patches_dir = Path(out_dir) / patch_dir_name
                pout_path = write_salient_patches(reader, Ai, used_spacing, out_dir=patches_dir,
                                                  patch_size=patch_size, shift=stride,
                                                  sal_result2=None, overwrite=self.overwrite)
                if pout_path is not None:#shouldnt happen
                    self.history_patch_pathes[name].append(pout_path)

                if self.visu_hmscaled:
                    if operation=='scale':
                        Ai = Ai-np.nanmin(Ai[Ai!=-np.inf])
                        if Ai.max()!=0:
                            Ai = Ai/Ai.max()
                    elif operation in ['sigm','sigmoid']:
                        Ai = expit(Ai)
                    Ai = (Ai*255).astype(np.uint8)

                hm_path = (hm_dir / (name + '_likelihood_map.tif')).absolute()
                self.history_hm_pathes[name].append(hm_path)
                write_wsi_heatmap(Ai, hm_path, used_spacing, wsi=reader, links_dir=hm_dir, shift=stride,
                                      anno_path=anno_path, overwrite=self.overwrite)
                self.history_hm_pathes[name].append(hm_path)

                hm_thumbs_dir = Path(out_dir) / ('hm_thumbs' + class_suffix)
                hm_thumbs_path, thumb_created = create_pred_thumbnail(mask_path=hm_path, wsi_path=reader, out_dir=hm_thumbs_dir,
                                                           heatmap=True, overwrite=self.overwrite, alpha=0.5, duo=True)
                self.history_hm_thumb_pathes[name].append(hm_thumbs_path)



    def save_hm_and_patches_links(self, out_dir, row, target_cols=None):
        if self.no_links:
            return
        # make true/wrong combination dir with links
        name = row[self.name_col]
        out_dir = Path(out_dir)
        for hmp in self.history_hm_pathes[name]:
            wsi_path = CompressedDataset.get_wsi_path(row)
            anno_path = self._get_anno_path(name)
            true_wrong_hm_dir = out_dir / (hmp.parent.name + '_links')
            make_asap_link(wsi_path, anno_path, hmp, true_wrong_hm_dir, relative='anno', exist_ok=True)
        for pp in self.history_patch_pathes[name]:
            true_wrong_pp_dir = out_dir / (pp.parent.name + '_links')
            mkdir(true_wrong_pp_dir)
            make_link(pp, true_wrong_pp_dir / pp.name, relative=True, exist_ok=True)
        for hp in self.history_hm_thumb_pathes[name]:
            true_wrong_hp_dir = out_dir / (hp.parent.name + '_links')
            mkdir(true_wrong_hp_dir)
            make_link(hp, true_wrong_hp_dir / hp.name, relative=True, exist_ok=True)

    def _get_anno_path(self, name):
        anno_path = None
        if self.anno_dir is not None and str(self.anno_dir) != 'None' and len(str(self.anno_dir)) > 1:
            anno_path = Path(self.anno_dir) / (name + '.xml')
            if not anno_path.exists():
                anno_path = None
        return anno_path

class DummyHeatmapWriter(HeatmapWriter):
    def __init__(self):
        super().__init__(npz_dir=None)
    def evaluate(*args, **kwargs):
        return

class ClfEvaluationHeatmapWriter(HeatmapWriter):
    def _evaluate(self, data_conf:DataConf, out_dir):
        #will be run per category, save heatmaps, patches and links

        n_correct = defaultdict(int) #per target label
        n_wrong = defaultdict(int) #per target label
        if out_dir is None:
            return

        for idx, row in data_conf.df.iterrows():
            #TODO handle the case when there are no targets (save all)
            save = True
            target = data_conf.get_row_labels(row)
            pred = data_conf.get_row_pred_labels(row)
            correct = target==pred
            max_label = np.max(target)
            n_correct[max_label]+=correct
            n_wrong[max_label]+=not correct
            if (correct and n_correct[max_label]>self.n_hm_correct_max) or \
                (not correct and n_wrong[max_label]>self.n_hm_wrong_max):
                save = False

            if save:
                self.save_hm_and_patches(row, out_dir=out_dir, target_cols=data_conf.target_cols)
                self.save_hm_and_patches_links(Path(out_dir) / 'hmp_links', target, pred, row,
                                               target_cols=data_conf.target_cols)

    def save_hm_and_patches_links(self, out_dir, target, pred, row, target_cols=None):
        if self.no_links:
            return
        # make true/wrong combination dir with links
        name = row[self.name_col]
        if not is_iterable(target):
            target = [target]
        if not is_iterable(pred):
            pred = [pred]
        # for multilabel assume large label value to be the more important one
        pred = sorted(pred, reverse=True)
        gt = sorted(target, reverse=True)
        true_wrong_name = ''
        for gti in gt:
            true_name = target_cols[gti]
            true_wrong_name += true_name + '_'
        for predi in pred:
            pred_name = target_cols[predi]
            true_wrong_name += '_' + pred_name
        true_wrong_name = true_wrong_name.replace('__', '-')
        true_wrong_dir = Path(out_dir) / (true_wrong_name + '_links')
        mkdir(true_wrong_dir)
        for hmp in self.history_hm_pathes[name]:
            wsi_path = CompressedDataset.get_wsi_path(row)
            anno_path = self._get_anno_path(name)
            true_wrong_hm_dir = true_wrong_dir / (hmp.parent.name + '_links')
            make_asap_link(wsi_path, anno_path, hmp, true_wrong_hm_dir, relative='anno', exist_ok=True)
        for pp in self.history_patch_pathes[name]:
            true_wrong_pp_dir = true_wrong_dir / (pp.parent.name + '_links')
            mkdir(true_wrong_pp_dir)
            make_link(pp, true_wrong_pp_dir / pp.name, relative=True, exist_ok=True)
        for hp in self.history_hm_thumb_pathes[name]:
            true_wrong_hp_dir = true_wrong_dir / (hp.parent.name + '_links')
            mkdir(true_wrong_hp_dir)
            make_link(hp, true_wrong_hp_dir / hp.name, relative=True, exist_ok=True)

