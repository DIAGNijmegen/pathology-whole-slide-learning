import getpass

import time, os, sys
from collections import Counter
from copy import deepcopy
from enum import Enum
import numpy as np
from wsilearn.utils.cool_utils import print_mp, print_mp_err
from wsilearn.dataconf import DataConf
from wsilearn.dl.np_transforms import NumpyToTensor
from wsilearn.utils.df_utils import print_df, df_duplicates_check
from wsilearn.utils.path_utils import PathUtils

print = print_mp
from sklearn.preprocessing import MultiLabelBinarizer

import torch
from torch.utils.data import Dataset

from pathlib import Path
import pandas as pd
from wsilearn.utils.cool_utils import ensure_dir_exists, is_iterable, copy_file, call_ls, hwc_to_chw

from wsilearn.compress.compressed_data import CompressedInfo, compressed_slide_from_ending

class CompressedDataset(Dataset):
    cache_col = 'cache_path'
    wsi_path_col_old = 'wsi_path'
    wsi_path_col_old2 = 'img'
    wsi_path_col = 'image'
    data_col = 'data'
    target_col = 'target' #can be used for training directly
    coords_col = 'coords'

    label_col = 'label' #for binary clf same as target, for multilabel clf the target labels

    @staticmethod
    def get_wsi_path(row):
        path_cols = [CompressedDataset.wsi_path_col,CompressedDataset.wsi_path_col_old,CompressedDataset.wsi_path_col_old2]
        tried_pathes = {}
        for pc in path_cols:
            wsi_path = row.get(pc,None)
            if wsi_path is None: continue
            if is_iterable(wsi_path):
                wsi_path = wsi_path[0]
                wsi_path = wsi_path.strip()
            tried_pathes[pc] = wsi_path
            if Path(wsi_path).exists():
                break
        if wsi_path is None or str(wsi_path).lower()=='none':
            for k,v in row.items():
                print(k,':',v)
            print('tried pathes (as user: %s):' % getpass.getuser(), tried_pathes)
            # if isinstance(row, pd.Series):
                # rstring = row.to_markdown()
                # with pd.option_context('display.max_rows', None,
                #                        'display.width', 1500,
                #                        'display.precision', 3,
                #                        'display.colheader_justify', 'left'):
                #     print(row)
            # else:
            #     print(row)
            raise ValueError('didnt find image path tried %s' % (str(path_cols)))

        return wsi_path

    def __init__(self, compressed_dirs, data_conf:DataConf, split, transform=None, cache_dir=None,
                 convert_f32=False, convert_f16=False, debug=False, flat=True):
        """ flat: if true, returns the individual patches otherwise in [hxw] shape.
        """
        #features are saved in format: (channels, rows, columns) - (channels, height, width)

        # self.df = self.ch.to_df()
        # self.class_ratios = self.config_helper.get_class_ratios()
        if not is_iterable(compressed_dirs):
            compressed_dirs = [compressed_dirs]
        compressed_pathes = PathUtils.list_pathes(compressed_dirs, ending='h5', ret='str')
        compressed_names = [Path(p).stem for p in compressed_pathes]
        dfcomp = pd.DataFrame(dict(name=compressed_names))
        df_duplicates_check(dfcomp, 'name', verbose=True, raise_error=True)
        cinfo = CompressedInfo(compressed_dirs)
        self.code_size = cinfo.df.code_size.values[0]
        # self.df = self._merge_config_compressed_ch(self.config_helper, cinfo)

        # self.require_distance_map = require_distance_map
        self.cache_dir = cache_dir
        if transform is None:
            transform = NumpyToTensor()
        self.transform = transform

        self.convert_f32 = convert_f32
        if self.convert_f32:
            print('loading compressed as fp32 (float32)')
        self.convert_f16 = convert_f16
        if self.convert_f16:
            print('loading compressed as fp16 (float16)')
        if convert_f32 and convert_f16:
            print('f16 and f32 set!!')
        self.flat = flat
        self.debug = debug

        self.cache_time = 0
        self.cache_count = 0
        self.transform_time = 0
        self.transform_count = 0
        self.load_time = 0
        self.load_count = 0

        self.data_conf = deepcopy(data_conf)
        dfn = pd.DataFrame(dict(name=compressed_names, path=compressed_pathes))
        # self.data_conf.merge(cinfo.df, cinfo.name_col)
        self.data_conf.merge(dfn, 'name')
        self.data_conf.select(split=split)

        print('create %s dataset with %d images, cdirs %s' % (split, len(self.data_conf.df), compressed_dirs))

        self.binarizer = None
        if self.data_conf.is_multilabel():
            class_inds = list(range(self.data_conf.count_target_cols()))
            self.binarizer = MultiLabelBinarizer(classes=class_inds)
            self.binarizer.fit(class_inds)

        if cache_dir:
            self.cache_dir = Path(cache_dir)
            ensure_dir_exists(cache_dir)
            self._cache_compressed()

    def _balanced_surv_weights(self, bins=4):
        # return self.balanced_weights_col(self.data_conf.get_surv_event_col())

        ###### put durations in bins, return in inverse bin density """
        # durations = self.data_conf.get_targets_surv_durations()
        # events = self.data_conf.get_targets_surv_events()
        # durations_ev = [durations[i] for i in range(len(events)) if events[i]]
        # hist, bin_edges = np.histogram(durations_ev, bins=bins, density=True) #bin-edges: array from min to max of length #bins+1
        # bin_edges[0] = 0
        # bin_edges = bin_edges[:-1]
        # val_bins = np.digitize(durations, bin_edges)
        # bin_ratios = np.bincount(val_bins)/len(val_bins) #the more values in a bin the higher
        # weights = []
        # for vbin in val_bins:
        #     weights.append(1/bin_ratios[vbin])

        ### equal amount of elements in each bin - uncensored
        durations = self.data_conf.get_targets_surv_durations()
        events = self.data_conf.get_targets_surv_events()
        durations_ev = [durations[i] for i in range(len(events)) if events[i]]

        bin_edges = np.percentile(durations_ev, np.linspace(0, 100, bins+1))[:-1]
        bin_edges[0] = 0
        # assign each value to a bin starting with 1
        val_bins = np.digitize(durations, bin_edges)
        bin_ratios = np.bincount(val_bins)/len(val_bins) #the more values in a bin the higher
        weights = []
        for vbin in val_bins:
            weights.append(1/bin_ratios[vbin])


        return torch.DoubleTensor(weights)

    def balanced_weights(self):
        if self.data_conf.is_surv():
            return self._balanced_surv_weights()

        N = len(self.data_conf)
        weights = []
        # class_ratios = self.config_helper.get_class_ratios()
        class_ratios = self.data_conf.get_label_ratios()
        print('class_ratios:', class_ratios)
        for idx, row in self.data_conf.df.iterrows():
            labels = self.data_conf.get_row_labels(row)
            if not is_iterable(labels):
                labels = [labels]
            ratios = [class_ratios[l] for l in labels]

            w = 1/min(ratios) #like in clam N/N_c (weights dont have to sum to 1)
            # easier to see the difference in weighting compared to
            # w = 1/(N*min(ratios))
            weights.append(w)
        print('balance weights min: %.1f, mean: %.1f, max: %.1f' % (np.min(weights), np.mean(weights), np.max(weights)))
        return torch.DoubleTensor(weights)

    def balanced_category_weights(self):
        cat_ratios = self.data_conf.get_category_ratios()
        print('category ratios:', cat_ratios)
        weights = []
        for idx, row in self.data_conf.df.iterrows():
            cat = self.data_conf.get_row_category(row)
            ratio = cat_ratios[cat]
            w = 1/ratio
            weights.append(w)
        print('balance category weights min: %.1f, mean: %.1f, max: %.1f' % (np.min(weights), np.mean(weights), np.max(weights)))
        return torch.DoubleTensor(weights)

    def balanced_weights_col(self, col):
        cratios = self.data_conf.df[col].value_counts(normalize=True)
        weights = []
        for idx, row in self.data_conf.df.iterrows():
            cat = row[col]
            ratio = cratios[cat]
            w = 1/ratio
            weights.append(w)
        print('balance %s weights min: %.1f, mean: %.1f, max: %.1f' % (col, np.min(weights), np.mean(weights), np.max(weights)))
        return torch.DoubleTensor(weights)

    def get_targets(self):
        # return self.df[ConfigHelper.target_col]
        return self.data_conf.get_targets()

    def __len__(self):
        return len(self.data_conf)

    def get_max_dim(self):
        raise ValueError('implement?')
        # return max(self.df.dim1.max(), self.df.dim2.max())

    def get_max_dims(self, inds):
        raise ValueError('implement?')
        # selection = self.df.iloc[inds]
        # return int(selection.dim1.max()), int(selection.dim2.max())

    def _cache_compressed(self):
        if self.cache_dir is None:
            return
        ensure_dir_exists(self.cache_dir)

        print('caching %d compressed to %s' % self.cache_dir)
        orig_dirs = []
        cached = []
        for idx, row in self.data_conf.df.iteritems():
            orig_path = row[CompressedInfo.path_col]
            orig_dir = Path(orig_path).parent
            if orig_dir not in orig_dirs:
                call_ls(orig_dir) #for chansey
                orig_dirs.append(orig_dir)
            cache_path = self.cache_dir / Path(orig_path).name
            if not cache_path.exists():
                start_time = time.time()
            copy_file(orig_path, cache_path)
            self.cache_time+=(time.time()-start_time)
            self.cache_count+=1
            cached.append(str(cache_path))
        self.data_conf.df[self.cache_col] = cached


    def read_distance_map(self, idx):
        raise ValueError('implement')
        # dm_path = self._get_featurized_path(self.df.dm_name[idx])
        # dm_path = self._cache_file(dm_path)
        # distance_map = np_load_monitored(dm_path)
        # return distance_map


    def __getitem__(self, idx):
        start_time = time.time()
        # row_old = self.df.iloc[idx]
        row = self.data_conf.df.iloc[idx]

        if self.cache_col in row:
            path_col = self.cache_col
        else:
            path_col = CompressedInfo.path_col
        if path_col not in row:
            raise ValueError('no column %s, only: %s' % (path_col, str(row.keys())))
        path = row[path_col]
        path = Path(path)
        compressed_slide = compressed_slide_from_ending(path.suffix, path=path)
        x, infos = compressed_slide.read(flat=self.flat)
        self.load_time += (time.time() - start_time)
        self.load_count += 1

        if compressed_slide.hwc and not self.flat:
            x = hwc_to_chw(x)

        if self.convert_f16 and x.dtype!=np.float16:
            x = x.astype('float16')
        if self.convert_f32 and x.dtype!=np.float32:
            x = x.astype('float32')

        x_shape = x.shape
        if self.transform:
            start_time = time.time()
            x = self.transform(x)
            self.transform_time+=(time.time()-start_time)
            self.transform_count+=1
            xtransf_shape = x.shape
            if idx==0:
                print('idx=0: returning %s transformed to %s' % (str(x_shape), str(xtransf_shape)))

        # #now back from hwc to original format
        # if not self.hwc: #hwc->chw
        #     x = hwc_to_chw(x)

        # y = self.df[ConfigHelper.target_col][idx]
        # y = torch.from_numpy(y)
        # rest_info = dict(self.df.iloc[idx])

        y = self.data_conf.get_row_target(row)

        row = dict(row)
        if not self.flat:
            infos.pop(self.coords_col)
        row.update(infos)#to have the shape as array
        # if self.coords_col in infos:
        #     row[self.coords_col] = infos.pop(self.coords_col)
        # rest_info.pop(self.target_col)

        # rest_info['dm_name']
        # d = DataAdapter().pack_inputs_labels_to_dict(x, y)
        # d.update(**rest_info)

        if x.isnan().any():
            print('compressed x has nans!')
        row.update({self.data_col:x, self.target_col:y})
        return row
        # return x, y, rest_info


def collate_compressed_dict(batch):
    collated = {}

    inputs = torch.cat([item[CompressedDataset.data_col] for item in batch], dim = 0)
    # if is_iterable(batch[0][CompressedDataset.target_col]):
    #     targets = np.array([item[CompressedDataset.target_col] for item in batch])
    #     targets = torch.cat(targets, dim = 0)
    # else:
    targets = torch.tensor([item[CompressedDataset.target_col] for item in batch])
    # label = torch.LongTensor([item.target for item in batch])
    collated[CompressedDataset.data_col] = inputs
    collated[CompressedDataset.target_col] = targets
    # if CompressedDataset.coords_col in batch[0]:
    #     coords = np.array([item[CompressedDataset.coords_col] for item in batch]).squeeze()
    #     collated[CompressedDataset.coords_col] = coords

    keys = list(batch[0].keys())
    for k in keys:
        if k in [CompressedDataset.data_col, CompressedDataset.target_col]:
            continue
        collated[k] = [item[k] for item in batch]
    return collated