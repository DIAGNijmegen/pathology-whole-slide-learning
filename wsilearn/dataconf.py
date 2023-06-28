from __future__ import annotations

import itertools
from collections import Counter, OrderedDict, defaultdict
from copy import deepcopy

from itertools import combinations
from pathlib import Path

import sklearn
from numpy.random import RandomState
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, KFold
from typing import Tuple, List, Union
import numpy
import pandas as pd
import torch
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer

import numpy as np

from wsilearn.utils.cool_utils import is_df, is_iterable, read_yaml_dict, is_list, is_ndarray, lists_flatten, is_string, \
    string_empty, is_int, mkdir, list_strip, list_intersection
from wsilearn.utils.df_utils import print_df, df_columns_starting_with, df_merge_check, df_object_col, df_delete_cols, \
    df_duplicates_check, df_merge, df_concat
from wsilearn.utils.sparse_utils import hoyer
from collections import OrderedDict

def _is_tensor(obj):
    return type(obj) is torch.Tensor


class TrainType(object):
    clf = 'clf'
    multilabel = 'multilabel'
    regr = 'regr'
    surv = 'surv'
    types = [clf, multilabel, regr, surv]
    clf_types = [clf, multilabel]

    @classmethod
    def post_fct(cls, train_type):
        if train_type==TrainType.clf:
            return 'softmax'
        elif train_type==TrainType.multilabel:
            return 'sigmoid'
        elif train_type==TrainType.surv:
            return 'surv'
        else:
            return None

    @classmethod
    def monitor(cls, train_type):
        if train_type in [TrainType.clf, TrainType.multilabel]:
            return 'val_auc', 'max'
        elif train_type==TrainType.surv:
            return 'val_cind', 'max'
        elif train_type==TrainType.regr:
            return 'val_loss', 'min'
        else:
            raise ValueError('unknown monitor')

    @classmethod
    def adapt_targets(cls, labels, train_type):
        if train_type not in TrainType.types:
            raise ValueError('unknown train type %s' % train_type)

        if is_list(labels):
            labels = np.array(labels)

        if is_ndarray(labels):
            if train_type in [TrainType.clf,TrainType.multilabel] and labels.dtype==np.float:
                labels = labels.astype(np.int64)
            elif train_type==TrainType.regr and labels.dtype != np.float:
                labels = labels.astype(np.float32)

        if _is_tensor(labels):
            if train_type in [TrainType.clf,TrainType.multilabel] and labels.dtype==torch.float32:
                labels = labels.long()
            elif train_type==TrainType.regr and labels.dtype != torch.float:
                labels = labels.float()

            if len(labels.shape)==1 and train_type==TrainType.regr: #since the model will return n_batchx1 for one-dimensional regression
                labels = labels.unsqueeze(-1)

        return labels

    @staticmethod
    def determine(targets):
        """ determines the TrainType, formatted targets and n_classes. start_labels_at_zero
        format: regr: targets directly
                clf,multilabel: one_hot
        n_classes: for clf: n_classes for clf; for the others the number of regr dimensions
        """
        if not is_iterable(targets[0]):
            targets = [[tar] for tar in targets]

        targets_lens = [len(tars) for tars in targets]
        ulens = list(set(targets_lens))

        if len(ulens)==1 and 1 in ulens: # one target per image
            targets = np.array([tar for tar in targets])
            if ((targets % 1)==0).all():
                targets = targets.astype(int)
                train_type = TrainType.clf
                targets-=min(targets)
                utargets = sorted(list(np.unique(targets)))
                n_dims = len(utargets)
                if np.max(utargets)+1!=n_dims:
                    raise ValueError('%d classes, but bad ordering: %s' % (n_dims, str(utargets)))
                target_onehot = np.zeros((len(targets),n_dims), dtype=np.uint8)
                for j,targ in enumerate(targets):
                    target_onehot[j,targ] = 1
                targets = target_onehot
            else:
                train_type = TrainType.regr
                n_dims = 1
        else:
            uvals  = np.array(lists_flatten(targets))
            if ((uvals % 1)  == 0).all(): #multilabel
                train_type = TrainType.multilabel
                targets = MultiLabelBinarizer().fit_transform(targets)
            else:
                if len(ulens)!=1:
                    raise ValueError('different number of non-integer targets! %s' % str(targets))
                train_type = TrainType.regr
                targets = np.array(targets)
            n_dims = targets.shape[1]

        return train_type, targets, n_dims

    @staticmethod
    def format(targets, train_type, start_labels_at_zero=True):
        if train_type == TrainType.clf:
            targets = np.array(targets).astype(np.int64)
            if start_labels_at_zero:
                targets-=targets.min()
        elif train_type == TrainType.multilabel:
            targets = MultiLabelBinarizer().fit_transform(targets)
        else:
            targets = np.array(targets).astype(np.float32)
        return targets

    @staticmethod
    def torch_logits_to_preds(train_type, logits):
        import torch.nn.functional as F
        if train_type==TrainType.clf:
            Y_hat = torch.topk(logits, 1, dim = 1)[1]
            Y_prob = F.softmax(logits, dim = 1)
        elif train_type==TrainType.multilabel:
            Y_prob = torch.sigmoid(logits)
            # Y_hat = Y_prob.data > 0.5
            Y_hat = Y_prob >= 0.5
        else: #regr
            return logits, logits
        return Y_prob, Y_hat

class DataConf(object):
    """ Data configuration for classification (multicvlass), multilabel classification, regression and survival
    For survival the target_names are in the order (event, duration)
    """
    name_col = 'name'
    split_col = 'split'
    category_col = 'category'

    image_col = 'image'
    mask_col = 'mask'

    target_col = 'target'
    out_col = 'out'
    label_col = 'label' #useful for clf (not multilabel)
    classes_col = 'classes' #for clf to have better readability of the table

    pred_col = 'pred'

    def __init__(self, data, train_type, target_names, split_col=split_col, category_col=category_col, name_col=name_col,
                 out_col=out_col, pred_col=pred_col, label_col=label_col, categories_order=None, default_category='default', check_duplicates=True):
        """ data: df or csv path """
        self.name_col = name_col
        self.split_col = split_col
        self.category_col = category_col
        self.label_col = label_col

        if is_string(target_names):
            target_names = target_names.split(',')
            target_names = [cl.strip() for cl in target_names]
        self.target_cols = target_names

        self.train_type = train_type
        # if class_names is None:
        #     raise ValueError('DataConf requires class_names')
        # if is_string(class_names):
        #     class_names = class_names.split(',')
        #     class_names = [cl.strip() for cl in class_names]
        self.class_names = target_names

        self.out_col = out_col
        self.pred_col = pred_col

        if is_df(data):
            df = data.copy()
        else:
            df = pd.read_csv(data, dtype={self.name_col:str})
        if not df_object_col(df, self.name_col):
            df[self.name_col] = df[self.name_col].astype(str)

        # if split_col not in df:
        #     df[split_col] = 'default'
        # if category_col not in df:
        #     df[category_col] = 'default'

        if split_col not in df:
            df[split_col] = default_category

        self.df = df.copy()

        self._preprocess_clf()

        if self.is_surv():
            assert self.get_targets_surv_events().max()==1

        if categories_order is not None and not is_iterable(categories_order):
            categories_order = categories_order.split(',')
            categories_order = list_strip(categories_order)
        self.categories_order = categories_order
        if categories_order is not None:
            print('categories order:', categories_order)

        df_duplicates_check(self.df, self.name_col, verbose=True, raise_error=check_duplicates)
        self.check_split_overlap()

    def check_split_overlap(self, name_col=None):
        if name_col is None: name_col = self.name_col
        splits = self.get_splits()
        combis = list(itertools.combinations(splits, 2))
        for split1, split2 in combis:
            names1 = self.df[self.df[self.split_col]==split1][name_col].unique()
            names2 = self.df[self.df[self.split_col]==split2][name_col].unique()
            inters = list_intersection(names1, names2)
            if len(inters)>0:
                print('Split intersection! %s and %s interesect in %d %s' % (split1, split2, len(inters), name_col))
                print(inters)

    @classmethod
    def like(self, data, data_conf:DataConf):

        return DataConf(data, train_type=data_conf.train_type, target_names=data_conf.target_cols,
                        categories_order=data_conf.categories_order,
                        split_col=data_conf.split_col, name_col=data_conf.name_col,
                        category_col=data_conf.category_col)

    def _preprocess_clf(self, df=None):
        if df is None:
            df = self.df
        df_duplicates_check(df, self.name_col)

        if self.is_clf_or_multilabel():
            tcols = self.get_target_cols()
            if len(tcols)==1: #binary or multiclass
                self.n_classes = len(self.get_targets_unique(df=df))
            else:
                self.n_classes = len(tcols) #multilabel hot-enc
        if self.is_clf_or_multilabel():
            self.create_classes()
        # self.create_labels(df=df, overwrite=False)

    def create_labels(self, df=None, overwrite=False, label_col=None):
        if label_col is None:
            label_col = self.label_col
        if df is None:
            df = self.df
        if self.is_clf_or_multilabel() and (label_col not in df or overwrite):
            labs = self.get_labels(df, overwrite=overwrite)
            if self.is_multilabel():
                labs = [' '.join(str(ls)) for ls in labs]
            df[label_col] = labs
        return df

    def create_classes(self, df=None, overwrite=False, classes_col=None):
        if classes_col is None:
            classes_col = self.classes_col
        if df is None:
            df = self.df
        if overwrite or classes_col not in df:
            df[classes_col] = self.get_classes(df=df)
        return df

    def remap_labels(self, label_map, order=None, delete_old_cols=True):
        if self.train_type not in TrainType.clf_types:
            raise ValueError('only for clf')

        lmap = {c:c for c in self.class_names}
        # lmap = {l:l for l in self.class_names}
        lmap.update(label_map)

        if list(lmap.keys())==list(lmap.values()):
            print('no remapping, since nothing would change')
            return

        #delete old label col
        if self.label_col in self.df:
            del self.df[self.label_col]

        dfn = self.df.copy()
        if delete_old_cols:
            df_delete_cols(dfn, self.class_names)

        if order is None:
            order = list(set((lmap.values())))
        for nn in order:
            dfn[nn] = 0
        for old,n in lmap.items():
            if old in self.df:
                dfn.loc[self.df[old]==1, n] = 1
        self.class_names = order
        self.target_cols = order

        self._preprocess_clf(dfn)
        self.df = dfn

    def delete_secondary_multilabels(self, label):
        """ deletes the given label if the sample has any other label """
        if not self.is_multilabel():
            raise ValueError('only for multilabel configs')
        labels = self.get_target_cols()
        if label not in labels: raise ValueError('unknown label %s' % label)
        other = [l for l in labels if l!=label]

        self.df.loc[(self.df[label]==1) & (self.df[other].sum(1)>=1), label] = 0

    def reorder_columns(self, cols, df=None):
        dfparam = False
        if df is None:
            dfparam = True
            df = self.df
        columns = list(df.columns.values)
        for c in cols:
            columns.remove(c)
        columns.extend(cols)
        df = df[columns].copy()
        if not dfparam:
            self.df = df
        return df

    def to_streaming(self, out_dir, prefix=''):
        """ converts to streaming format (three files) """
        binary_col = self.target_cols[-1]
        if len(self.target_cols)>2:
            raise ValueError('todo multiclass streaming config')
        if len(prefix)>0:
            prefix +='_'
        for split in ['training','validation','testing']:
            df = self.df.copy()
            df = df[df.split==split]
            df = df.rename(columns=dict(name='slide'))
            df = df.rename(columns={binary_col:'label'})
            out_path = Path(out_dir)/(prefix+split+'.csv')
            df[['slide','label']].to_csv(str(out_path), index=None)
            print('%s' % str(out_path))

    def is_clf(self):
        return self.train_type==TrainType.clf

    def is_binary(self):
        return self.train_type==TrainType.clf and len(self.class_names)==2
    def is_multiclass(self):
        return self.train_type==TrainType.clf and len(self.class_names)>2

    def is_multilabel(self):
        return self.train_type==TrainType.multilabel

    def is_regr(self):
        return self.train_type == TrainType.regr

    def is_surv(self):
        return self.train_type == TrainType.surv

    def is_clf_or_multilabel(self):
        return self.is_clf() or self.is_multilabel()

    def _one_hot(self, labels_col):
        df = self.df

        cnames = self.class_names
        for cname in cnames:
            vals = []
            for idx, row in df.iterrows():
                labels = row[labels_col]
                if string_empty(labels):
                    val = labels
                if not is_iterable(labels):
                    for sep in [',','|']:
                        if sep in labels:
                            labels = labels.split(sep)
                            break
                if not is_iterable(labels):
                    labels = [labels]
                elif cname in labels:
                    val = 1
                else:
                    val = 0
                vals.append(val)
            df[cname] = vals
        return df

    def delete_categories(self, categories):
        if self.category_col not in self.df:
            raise ValueError('no categories defined')
        if not is_iterable(categories):
            categories = [categories]
        self.df = self.df[~self.df[self.category_col].isin(categories)].copy()

    def rename_categories(self, rename_map):
        for old,n in rename_map.items():
            self.df.loc[self.df[self.category_col]==old, self.category_col] = n

    def get_categories(self, n=''):
        if self.category_col+str(n) not in self.df:
            return []
        cats = sorted(list(self.df[self.category_col+str(n)].unique()))
        if self.categories_order is not None and len(str(n))==0: #only for primary categories
            if set(self.categories_order) != set(cats) or len(cats)!=len(self.categories_order):
                raise ValueError('categories_order %s different from actual categories %s' \
                                 % (str(self.categories_order), str(cats)))
            return self.categories_order
        return cats

    def get_categories_sequence(self, df=None, n=''):
        if df is None:
            df = self.df
        return df[self.category_col+str(n)].values

    def get_splits(self):
        if self.split_col not in self.df:
            return []
        splits = sorted(list(self.df[self.split_col].unique()))
        return splits

    def sample(self, n=20):
        if self.is_multilabel():
            raise ValueError('todo')
        nx = int(np.round(n / (len(self.get_categories())*len(self.get_splits())*len(self.get_targets_unique()))))
        if nx<1: raise ValueError('n=%d to small' % n)
        df_samples = []
        for lab in self.get_targets_unique():
            for split in self.get_splits():
                for cat in self.get_categories():
                    dfs = self.df[(self.df[self.split_col] == split) & (self.df[self.category_col] == cat) & (self.df[self.target_col] == lab)]
                    if len(dfs)>0:
                        df_samples.append(dfs.sample(nx))
        dfs = pd.concat(df_samples)
        return DataConf(dfs, train_type=self.train_type, target_names=self.class_names)

    def select(self, df=None, split:Union[str, list, tuple]=None, category:Union[str, list, tuple]=None,
               names=None, category_n=''):
        """ removes data not in the given split and category"""
        ret_df = df is not None
        if df is None:
            df = self.df
        if split is not None:
            if not is_iterable(split):
                split = [split]
            df = df[df[self.split_col].isin(split)]
        if category is not None:
            if not is_iterable(category):
                category = [category]
            df = df[df[self.category_col+str(category_n)].isin(category)]
        if names is not None and len(names)>0:
            df = df[df[self.name_col].isin(names)].copy()
        if ret_df:
            return df.copy()
        else:
            self.df = df.copy()

    def get_slide_names(self, df=None, category=None):
        if not is_iterable(category):
            category = [category]
        if df is None:
            df = self.df
        df = df[df[self.category_col].isin(category)]
        return list(df[self.name_col].values)

    def path_for_name(self, name):
        dfr = self.df[self.df[self.name_col]==name]
        if len(dfr)==0:
            return None
        return dfr[self.image_col].values[0]

    @staticmethod
    def concat(confs):
        conf0 = confs[0]
        dfs = [conf.df for conf in confs]
        df = pd.concat(dfs, ignore_index=True)
        conf = deepcopy(conf0)
        conf.df = df
        return conf

    def merge(self, df, name_col):
        """ merges the given df with self on name_col """
        # dfc.rename(columns={ch.path_col:self.wsi_path_col}, inplace=True)
        # dfi = cinfo.df
        if not df_object_col(df, name_col):
            df[name_col] = df[name_col].astype(str)

        try:
            dfm, dfsmi, dfmi = df_merge_check(self.df, df, left=self.name_col, right=name_col,
                                              left_title='self.df', right_title='other df', suffixes=('__x', '__y'))
            cols = [str(c) for c in dfm.columns]
            failed_cols = []
            for c in cols:
                if c.endswith('__x') and c.replace('__x','__y') in cols:
                    failed_cols.append(c)
            if len(failed_cols)>0:
                print('DataConf-df:')
                print_df(self.df.head())
                print('other df:')
                print_df(df.head())
                raise ValueError(f'after merging df with columns {str(self.df.columns)} '
                                 f'with df with columns {str(self.df.columns)}, '
                                 f'left on {self.name_col}, right on {name_col}, '
                                 f'{len(failed_cols)} duplicate columns: {str(failed_cols)}')
        except:
            print('dataconf-merge with a df failed')
            print('dataconf columns:', self.df.columns)
            print('df columns:', df.columns)
            raise
        if len(dfm)!=len(self.df):
            df_merge_check(self.df, df, verbose=True, left=self.name_col, right=name_col,
                           left_title='config', right_title='compressed')
            raise ValueError('%d slides in config, but %d after merging!' % (len(self.df), len(dfm)))
        self.df = dfm
        return df

    def add(self, conf, same_cols=False):
        if is_df(conf):
            other_df = conf
        else:
            other_df = conf.df
        if same_cols:
            cols = sorted([str(c) for c  in self.df.columns])
            other_cols = sorted([str(c) for c in other_df.columns])
            if set(cols)!=set(other_cols):
                raise ValueError('adding conf conflict, different cols: \n%s and \n%s' % (str(cols), str(other_cols)))
        else:
            cols = [self.category_col, self.split_col, self.name_col]+self.get_target_cols()
            self_cols = list(self.df.columns.values)
            other_cols = list(conf.df.columns.values)
            if not set(cols).issubset(set(self_cols)):
                raise ValueError('adding conf conflict, self-conf missing cols: \n%s and \n%s' % (str(cols), str(self_cols)))
            if not set(cols).issubset(set(other_cols)):
                raise ValueError('adding conf conflict, other-conf missing cols: \n%s and \n%s' % (str(cols), str(self_cols)))
        self.df = pd.concat((self.df, other_df), ignore_index=True)

    @staticmethod
    def from_config(path, **kwargs):
        path = Path(path)
        if path.suffix.endswith('yaml'):
            return DataConf.from_yaml(path, **kwargs)
        elif path.suffix.endswith('csv'):
            return DataConf(path, **kwargs)
        else:
            raise ValueError('unknown suffix %s' % path.suffix)

    @staticmethod
    def from_yaml(path, target_names, train_type=None, split_col=split_col, category_col=category_col, name_col=name_col,
                  target_col=target_col, image_col=image_col, mask_col=mask_col,
                  ignore_pathes=False, **kwargs):
        """ returns 2 dictionaries: {purpose->[pathes]}, {path->mask} """
        if not Path(path).exists():
            raise ValueError('confing %s doesnt exist' % path)
        conf_map = read_yaml_dict(path)
        path_keys = conf_map['path']

        entries = []
        data_map = conf_map['data']
        default_split_name_map = {'train':'training', 'val':'validation', 'valid':'validation',
                                  'test':'testing'}
        for split, cat_map in data_map.items():
            split = default_split_name_map.get(split, split)
            for cat, vals in cat_map.items():
                for val in vals:
                    entry = {split_col:split, category_col:cat}
                    img = val.pop(image_col)
                    mask = val.pop(mask_col,'')
                    for k,v in path_keys.items():
                        k = '{%s}' % k
                        img = img.replace(k,v)
                        mask = mask.replace(k,v)
                    name = Path(img).stem
                    targets = val.pop('labels')
                    entry[target_col] = targets
                    if not ignore_pathes:
                        entry.update(dict(img=img, mask=mask))
                    if len(mask)==0:
                        del entry['mask']
                    entry[name_col] = name
                    entry.update(val)
                    entries.append(entry)

        df = pd.DataFrame(entries)
        targets_lists = df[target_col].values

        #add target cols
        train_type_, targets, n_dims = TrainType.determine(targets_lists)
        if train_type is None:
            train_type = train_type_
        if target_names is None:
            target_names = ['cl' + str(i) for i in range(n_dims)]
        elif is_string(target_names):
            target_names = target_names.split(',')
            target_names = [cl.strip() for cl in target_names]

        for d in range(n_dims):
            df[target_names[d]] = targets[:, d]

        # if len(targets.shape)==2 and targets.shape[1]>2:
        #     target_cols = []
        #     for j in range(targets.shape[1]):
        #         target_col = target_col+str(j)
        #         df[target_col] = targets[j]
        #         target_cols.append(target_col)
        # else:
        #     target_cols = [target_col]
        #     df[target_col] = targets
        # target_cols = list(sorted(target_cols))
        # return DataConf(df, train_type=train_type, target_cols=target_cols, class_names=class_names)
        del df[target_col]
        return DataConf(df, train_type=train_type, target_names=target_names, **kwargs)


    def count_target_cols(self):
        return len(self.target_cols)

    def get_target_cols(self):
        return self.target_cols
        # """ target_col is the default, if there are additional (target0,...) returns those only """
        # tcols = df_columns_starting_with(self.df, self.target_cols)
        # if len(tcols)>1: #onehot
        #     tcols.remove(self.target_cols)
        # return tcols

    def get_surv_event_col(self):
        assert self.is_surv()
        return self.target_cols[0]
    def get_surv_duration_col(self):
        assert self.is_surv()
        return self.target_cols[1]

    def get_targets_surv_events(self, df=None):
        assert self.is_surv()
        return self.get_targets(df)[:,0]

    def get_targets_surv_durations(self, df=None):
        assert self.is_surv()
        return self.get_targets(df)[:,1]

    def get_targets(self, df=None):
        if df is None:
            df = self.df
        tcols = self.get_target_cols()
        return df[tcols].values

    def get_targets_labels(self, df=None):
        if not self.is_clf():
            raise ValueError('only for clf')
        if self.is_multilabel():
            raise ValueError('implement returning list of lists with labels')
        #multiclass
        tars = self.get_targets(df=df)
        labs = tars.argmax(1)
        return labs


    def get_tagets_hot(self, df=None):
        if not self.is_clf_or_multilabel():
            raise ValueError('targets_hot only for clf and multilabel')

        if df is None:
            df = self.df
        targets = self.get_targets(df)
        if self.is_multilabel():
            return targets #already one-hot
        targets = MultiLabelBinarizer().fit_transform(targets)
        return targets

    def is_multitarget(self):
        return len(self.get_target_cols())>1

    def get_targets_unique(self, df=None):
        target_values = self.get_targets(df=df)
        if self.is_clf_or_multilabel():
            values = []
            for targ in target_values:
                vals = np.where(targ==1)[0]
                values.extend(list(vals))
            values = list(sorted(set(values)))
        else:
            values = sorted(list(set(target_values.ravel())))
        return values

    def has_multiple_categories(self):
        return self.get_categories() is not None and len(self.categories)>1

    def is_multicategory(self):
        return self.has_multiple_categories()

    def has_categories(self, n=''):
        return len(self.get_categories(n=n))>1 #if more then the one (default) category

    def has_splits(self):
        return len(self.get_splits())>0

    def print(self, *args, **kwargs):
        return print_df(self.df, *args, **kwargs)

    def get_category_groups(self, n=''):
        """ The data categories can be evaluable - contain all labels, or not evaluable - only single label.
        returns all evaulable categories and category combinations
         :return list of category groups ([[cat1,cat2],[cat1,cat3]])
         """
        categories = self.get_categories(n=n)
        if len(categories)==1:
            return [categories]

        groupings = []
        for r in range(1, len(categories)):
            groupr = list(set(combinations(categories, r)))
            groupr = [sorted(list(gro)) for gro in groupr]
            groupings.extend(sorted(groupr))

        if self.train_type in TrainType.clf_types:
            sel_groupings = []
            df = self.df
            for grouping in groupings:
                dfc = df[df[self.category_col+str(n)].isin(grouping)]
                if len(dfc)==0: continue
                glabel_values = self.get_targets_unique(dfc)
                if len(glabel_values)==self.n_classes:
                    sel_groupings.append(grouping)
            groupings = sel_groupings

        return groupings

    def _select_split(self, split=None):
        if split in [None,'all','All']:
            split = self.get_splits()
        elif not is_iterable(split):
            split = [split]
        df = self.df[self.df[self.split_col].isin(split)]
        return df

    def get_label_ratios(self, split=None, normalized=True):
        df = self._select_split(split=split)
        targets = self.get_targets(df)
        if self.is_clf():
            assert len(targets.shape)==2 and targets.shape[1]>1
            targets = targets.argmax(1)
            counter = Counter(targets.ravel())
        elif self.is_multilabel():#onehot
            sums = targets.sum(axis=0)
            counter = dict(zip(range(len(sums)), sums))
        else:
            raise ValueError('no classes for type %s' % self.train_type)

        results = OrderedDict()
        total = np.sum(list(counter.values()))
        for k in sorted(counter.keys()):
            val = counter[k]
            if normalized:
                val = val/total
            results[k] = val

        return results

    def get_category_ratios(self, split=None, normalized=True):
        df = self._select_split(split=split)
        cats = self.get_categories_sequence(df)
        counter = dict(Counter(cats))
        if normalized:
            counter = {k:v/len(df) for k,v in counter.items()}
        return counter

    def get_split_count(self, split=None, df=None):
        if df is None: df = self.df
        count = df.groupby(self.split_col).count()
        if split is not None:
            count = count[split]
        return count

    def info(self):
        df = self.df
        print('%d entries, %d columns:' % (len(df), len(df.columns)), list(df.columns))
        assert self.df[self.name_col].nunique()==len(self.df), 'non unique names'

        abs_count = dict(df.groupby(self.split_col).count()[self.name_col])
        dfc = df.groupby(self.split_col).count()
        rel_count = dict(dfc.name / sum(dfc.name))

        n = len(df)
        entries = []
        print('%d splits:' % len(abs_count))
        for split, scount in abs_count.items():
            print('%s: %.2f (%d)' % (split, rel_count[split], scount))
            entries.append(dict(split=split, category='all', labels='all', n=scount, ratio=scount/n))
        if self.get_categories() is not None and len(self.get_categories())>0:
            print('by category:')
            cat_count_map = dict(df.groupby(self.category_col).count()[self.name_col])
            for cat, ccount in cat_count_map.items():
                entries.append(dict(split='all', category=cat, labels='all', n=ccount, ratio=ccount/n))
            print('by purpose and category:')
            prup_cat_count_map = dict(df.groupby([self.split_col, self.category_col]).count()[self.name_col])
            for (purp, cat), pccount in prup_cat_count_map.items():
                entries.append(dict(split=purp, category=cat, labels='all', n=pccount, ratio=pccount/n))
            if self.is_clf():
                print('by purpose, category and label:')
                purp_cat_lab_count_map = dict(df.groupby([self.split_col, self.category_col, self.label_col]).count()[self.name_col])
                for (purp, cat, lab), pccount in purp_cat_lab_count_map.items():
                    entries.append(dict(split=purp, category=cat, labels=lab, n=pccount, ratio=pccount/n))
            print_df(pd.DataFrame(entries))

            if 'training' in df[self.split_col] and 'validation' in df[self.split_col]:
                df = df.copy()
                df.loc[df[self.split_col]=='training',self.split_col] = 'train_valid'
                df.loc[df[self.split_col]=='validation',self.split_col] = 'train_valid'
                print('by purpose train+valid and category:')
                print(dict(df.groupby([self.split_col, self.category_col]).count()[self.name_col]))
                if self.is_clf():
                    print(dict(df.groupby([self.split_col, self.category_col, self.label_col]).count()[self.name_col]))

        info_splits = self.get_splits()
        if len(info_splits)>1:
            info_splits.insert(0, 'all')
        for purp in info_splits:
            print('%s class ratios' % purp)
            print([f"{k}:{v:.2f} ({self.get_label_ratios(split=purp, normalized=False)[k]:d})"
                   for k,v in self.get_label_ratios(split=purp).items()])
            # print('%s class ratios abs:' % purp)
            # print([f"{k}:{v}" for k,v in self.get_label_ratios(split=purp, normalized=False).items()])

    def get_out_cols(self):
        out_cols = df_columns_starting_with(self.df, self.out_col)
        return out_cols

    def get_out(self):
        out_cols = self.get_out_cols()
        preds = self.df[out_cols].values
        return preds

    def get_row_target(self, row):
        tcols = self.get_target_cols()
        if len(tcols)==1:
            return row[tcols[0]]
        else:
            return np.array([row[tc] for tc in tcols])

    def get_row_labels(self, row):
        tcols = self.get_target_cols()
        return self._get_row_labels(row, tcols)

    def get_row_category(self, row):
        return row[self.category_col]

    def _get_row_labels(self, row, tcols):
        if self.is_clf() or self.is_multilabel():
            if not is_iterable(tcols) or len(tcols)==1:
                raise ValueError('expects one-hot encoding, not single label')

            labs = []
            for t,tcl in enumerate(tcols):
                if row[tcl]:
                    labs.append(t)
            return labs
        else:
            raise ValueError('only for clf, not for %s' % self.train_type)


    def add_pred(self, pred):
        if is_list(pred):
            pred = np.array(pred)

        if self.is_clf() and (len(pred.shape)==1 or pred.shape[1]==1):
            pred = MultiLabelBinarizer(classes=range(len(self.class_names))).fit_transform(pred.ravel()[:,None])

        if len(pred.shape)==1 or pred.shape[1]==1:
            pred = pred.ravel()
            self.df[self.pred_col] = pred
        else:
            for c in range(pred.shape[1]):
                self.df[self.pred_col+str(c)] = pred[:,c]

        if self.is_clf_or_multilabel() and 'wrong' not in self.df:
            targ = self.get_targets()
            wrong = (targ!=pred).sum(1)>0
            self.df['pred_wrong'] = wrong.astype(int)
        if self.is_clf_or_multilabel() or self.is_regr():
            out = self.get_out()
            out_diff = np.abs(targ-out).sum(1)/self.n_classes
            self.df['pred_diff'] = out_diff

    def get_pred_cols(self, df=None):
        if df is None:
            df = self.df
        return df_columns_starting_with(df, self.pred_col, not_starting='pred_')

    def get_pred(self, df=None):
        if df is None:
            df = self.df
        cols = self.get_pred_cols(df)
        return df[cols].values

    def get_pred_labels(self, df=None):
        preds = self.get_pred(df)
        if self.is_multilabel() or not self.is_clf():
            raise ValueError('implement returning multilabel pred labels')
        preds = preds.argmax(1)
        return preds

    def get_row_pred_labels(self, row):
        pcols = self.get_pred_cols()
        return self._get_row_labels(row, pcols)

    def has_pred(self):
        return len(self.get_pred_cols())>0

    def __len__(self):
        return len(self.df)

    def to_csv(self, path, clear_label=False, cols=None):
        Path(path).parent.mkdir(exist_ok=True)
        df = self.df.sort_values(self.name_col)
        if clear_label and self.label_col in df:
            del df[self.label_col]
        if cols is not None:
            df = df[cols]
        df.to_csv(str(path), index=None)
        print(str(path))

    def save(self, path, **kwargs):
        return self.to_csv(path, **kwargs)

    def create_cv_splits_category(self, df=None, category_col=None, **kwargs):
        """ returns list of DataConfs """
        if df is None:
            df = self.df
        if category_col is None:
            category_col = self.category_col

        cats = df[category_col].unique()

        confs_per_split = defaultdict(list)
        for cat in cats:
            dfc = df[df[category_col]==cat]
            csplits = self.create_cv_splits(df=dfc, **kwargs)
            for i,iconf in enumerate(csplits):
                confs_per_split[i].append(iconf)
        fconfs = []
        for fold,confs in confs_per_split.items():
            fconf = DataConf.concat(confs)
            fconfs.append(fconf)
        return fconfs

    def get_labels(self, df=None, label_col=None, overwrite=False):
        if label_col is None:
            label_col = self.label_col

        if label_col not in df or overwrite:
            if self.train_type==TrainType.clf:
                targets = self.get_targets(df)
                labels = targets.argmax(1)
                assert len(labels)==len(df)
                return labels
            elif self.train_type == TrainType.multilabel:
                labels = []
                for idx, row in df.iterrows():
                    rlabs = []
                    for clind, label in enumerate(self.class_names):
                        if row[label]:
                            # rlabs.append(label)
                            rlabs.append(clind)
                    labels.append(rlabs)
            else:
                raise ValueError('get_labels only works for clf, not for', self.train_type)
        else:
            labels = df[label_col].values

        return labels

    def get_classes(self, df=None):
        if self.train_type not in [TrainType.clf, TrainType.multilabel]:
            raise ValueError('get_classes works only for classification, not %s' % self.train_type)

        classes = []
        for idx, row in df.iterrows():
            rlabs = []
            for clind, clname in enumerate(self.class_names):
                if row[clname]:
                    rlabs.append(clname)
            classes.append(' '.join(rlabs))

        return classes

    def pop_names(self, dfr, name_col='name', id_col='pa'):
        """ returns the df with the names and removes entries by the given id_col """
        df = self.df[self.df[name_col].isin(dfr[name_col].unique())].copy()
        self.df = self.df[~self.df[id_col].isin(df[id_col])].copy()
        return df


    def create_cv_splits(self, df=None, id_col=None, label_col=None, random_state=1, stratified=True,
                         splits=5, train_split='training', val_split='validation',
                         balance_cols=[], iterations=100, balance_fct=hoyer, verbose=False):
        """ with balance_cols samples iteration-times and takes the sample with the smallest balance_fct-value
        (least sparseness if hoyer) leading to a more balanced value distribution """
        if df is None:
            df = self.df
        if id_col is None:
            id_col = self.name_col
        print('splitting %d entries on %d %s in %d splits' % (len(df), df[id_col].nunique(), id_col, splits))
        assert self.train_type in TrainType.clf_types, 'only for clf'

        if label_col is None:
            label_col = self.label_col


        if self.train_type==TrainType.clf:
            self.create_labels(df=df, label_col=label_col)
        elif self.train_type==TrainType.multilabel:
            # df = self.create_labels(df=df, overwrite=True, label_col='labels')
            df = self.create_worst_multilabel_order(self.target_cols, df=df)
        else:
            raise ValueError('only tested for clf')
        # labels = self.get_labels(df, label_col)
        # dfsel[label_col] = labels
        dfsel = df.copy()
        dfsel = dfsel.drop_duplicates(id_col)
        # if len(dfsel)!=dfsel[id_col].nunique():
        #     raise ValueError('different labels for same id_col %s' % id_col)

        # print('splitting %d entries in %d splits' % (len(dfsel), splits))

        if is_int(random_state):
            random_state = RandomState(random_state)

        if stratified:
            kf = StratifiedKFold(n_splits=splits, random_state=random_state, shuffle=True)
        else:
            kf = KFold(n_splits=splits, random_state=random_state, shuffle=True)

        best_balance = 1; best_confs = None
        for i in range(iterations):
            confs = []; balances = []
            for train_index, test_index in kf.split(dfsel[id_col], y=dfsel[label_col]):
                train_ids = dfsel.iloc[train_index][id_col].values
                test_ids = dfsel.iloc[test_index][id_col].values
                for bcol in balance_cols:
                    balances.append(balance_fct(dfsel[dfsel[id_col].isin(test_ids)][bcol].dropna()))
                conf = deepcopy(self)
                conf.df = df.copy()
                # conf = DataConf(df.copy(), train_type=self.train_type, class_names=self.class_names)
                conf.df.loc[conf.df[id_col].isin(train_ids),self.split_col] = train_split
                conf.df.loc[conf.df[id_col].isin(test_ids),self.split_col] = val_split
                confs.append(conf)

            for j in range(1,len(confs)):
                conf = confs[j]
                lconf = confs[j-1]
                assert (conf.df[conf.df[self.split_col]==val_split][self.name_col].isin(lconf.df[lconf.df[self.split_col]=='training'][self.name_col])).all()

            #repeat to reach best balance of test set
            if balance_cols is None or len(balance_cols)==0:
                best_confs = confs
            else:
                balance = np.mean(balances)
                if balance<=best_balance:
                    best_confs = confs
                    best_balance = balance
                    if verbose:
                        print('best balance: %.5f' % balance)

        return best_confs

    def create_cv_splits_nested(self, df=None, id_col=None, label_col=None, random_state=1, stratified=True,
                             splits=5, train_split='training', val_split='validation', test_split='testing',
                             **kwargs)->dict:
        """ creates a dictionary name->conf, the name e.g. fold0_fold0 """
        conf_map = {}
        confs = self.create_cv_splits(df=df, id_col=id_col, label_col=label_col, random_state=random_state, stratified=stratified,
                                      splits=splits, train_split=train_split, val_split=test_split, **kwargs)
        #confs for cv3: [#train#train#test,#train#test#train#,test#train#train]
        print('creating subsplits...')
        for s,conf in enumerate(confs):
            train_conf = deepcopy(conf)
            train_conf.select(split=train_split) #train#train
            test_conf = deepcopy(conf) #test
            test_conf.select(split=test_split)
            train_confs = train_conf.create_cv_splits(id_col=id_col, label_col=label_col, random_state=random_state, stratified=stratified,
                                            splits=splits, train_split=train_split, val_split=val_split, **kwargs)
            #train_confs: [#train#val,#val#train]
            for f,trc in enumerate(train_confs):
                trc.add(test_conf)
                conf_map[f'fold{s}_fold{f}'] = trc
        #train#val#test, #val#train#test
        #train#test#val,#val#train#test
        #test#train#val,#test#val#train
        return conf_map

    def create_cv_splits_val_test(self, df=None, id_col=None, label_col=None, random_state=1, stratified=True,
                             splits=5, train_split='training', val_split='validation', test_split='testing',
                             **kwargs)->dict:
        """ creates n splits, eg. [tr,tr,val,te]
                                  [tr,val,te,tr]
                                  [val,te,tr,tr]
                                  [te,tr,tr,val]
        """
        if df is None: df = self.df
        if id_col is None: id_col = self.name_col

        confs = self.create_cv_splits(df=df, id_col=id_col, label_col=label_col, random_state=random_state, stratified=stratified,
                                      splits=splits, train_split=train_split, val_split=test_split, **kwargs)
        dfs = []
        for i, conf in enumerate(confs):
            other_conf = confs[i+1 if (i+1) < len(confs) else 0] #next conf
            #set in this conf the slides, which are test in the other conf as val
            conf.df.loc[conf.df[id_col].isin(other_conf.df[other_conf.df[DataConf.split_col]==test_split][id_col]),
                        DataConf.split_col] = val_split
            dfs.append(conf.df)
        dfs = df_concat(*dfs)
        dfsval = dfs[dfs[DataConf.split_col]==val_split]
        assert len(dfsval)==len(df)
        assert len(dfs[dfs[DataConf.split_col]==test_split])==len(df)
        if 'name' in dfs:
            assert dfsval['name'].nunique()==df['name'].nunique()
        return confs

    def create_worst_multilabel_order(self, order, df=None, label_col=None):
        """ orders the columns in the given order and sets the worst (last) label in the label_col """
        if df is None:
            df = self.df
        if label_col is None:
            label_col = self.label_col
        df = self.reorder_columns(order, df=df)
        #find worst labels
        max_labels = []
        label_arr = df[order].values
        for i,r in enumerate(label_arr):
            max_labels.append(r.nonzero()[0].max())
        df[label_col] = max_labels
        return df
        # print_df(dconf.df.head(20))


def convert_streaming_config(train_csv, valid_csv, target_names, test_csv=None, out_path=None, label_col='label', slide_dir=None):
    # df['ssplit'] = '?'
    dfs = []
    for split,path in (('training',train_csv),('validation',valid_csv), ('testing',test_csv)):
        if path is None:
            continue
        df = pd.read_csv(path)
        df['split'] = split
        df = df.rename(columns={'val':'name', 'train':'name', 'test':'name'})
        df['name'] = df['name'].str.replace('.tif','')
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    if slide_dir is not None:
        df['image'] = slide_dir+'/'+df['name']+'.tif'
    vals = df[label_col].unique()
    for val in vals:
        val = int(val)
        df[target_names[val]] = 0
        df.loc[df[label_col]==val, target_names[val]] = 1
    df[label_col] = df[label_col].astype(int)

    conf = DataConf(df, train_type=TrainType.clf, target_names=target_names)
    conf.info()
    if out_path is not None:
        mkdir(Path(out_path).parent)
        conf.to_csv(out_path)
    return df


if __name__ == '__main__':
    # dc = DataConf(path, target_names=['IDC','ILC'], train_type='clf')
    pass