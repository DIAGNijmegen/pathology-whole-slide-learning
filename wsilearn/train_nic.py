import os, getpass, sys
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers, seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, StochasticWeightAveraging
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler
from pytorch_lightning.trainer.states import TrainerStatus

from wsilearn.utils.df_utils import print_df
from wsilearn.att_net import InstClusterLoss

print('nic version 0.2', flush=True)

from wsilearn.wsi.slide_summary import SlidesInfo
from wsilearn.nic_utils import create_nic_model, DummyHeatmapWriter, ClfEvaluationHeatmapWriter, HeatmapWriter

#only for debugging
# os.environ['CUDA_LAUNCH_BLOCKING']="1"

import functools
import shutil
import time
from collections import OrderedDict, defaultdict
import pandas as pd
import numpy as np
import os

import scipy
from pathlib import Path


from wsilearn.utils.cool_utils import set_seed, read_json_dict, dict_of_lists_to_dicts, write_json_dict, is_string, \
    is_iterable, timer, time_to_str, mkdir, save_arrays, Timer, create_tensorboard_start_text, dict_add_key_prefx
from wsilearn.dl.models.model_utils import weights_init, WeightsInit
from wsilearn.dl.np_transforms import NumpyToTensor, RandomFlip, CenterCropPad, RandomRotate, RandomCropPad
from wsilearn.dl.pl.pl_logger import EpochCsvLogger, HistoryPlotCallback, MeterlessProgressBar
from wsilearn.dl.pl.pl_modules import ClfModule, RegrModule, ClfGroupLoss
from wsilearn.dl.pl.pl_inference import inference_trainer, load_pl_state_dict, Inferencer, InferenceCallback, \
    InferenceOutWriter, find_pl_model_path
from wsilearn.dl.torch_utils import to_numpy, create_device, print_model_summary
import torch
import torchvision.transforms as transforms

from wsilearn.utils.docker_utils import is_docker
from wsilearn.utils.eval_utils import ClfEvaluator, create_short_info_file, RegrEvaluator
from wsilearn.utils.survival_utils import SurvEvaluator
from wsilearn.dataconf import TrainType, DataConf
from wsilearn.utils.flexparse import FlexArgumentParser
from wsilearn.utils.io_utils import cache_dirs
from wsilearn.utils.path_utils import PathUtils

import matplotlib.pyplot as plt

from wsilearn.compress.compressed_dataset import CompressedDataset, collate_compressed_dict

print = functools.partial(print, flush=True)

def _create_valid_transforms():
    # valid_transforms = transforms.Compose([NumpyToTensor()])
    valid_transforms = NumpyToTensor()
    return valid_transforms

def create_train_transforms(crop_size=None, wiggle=0.05, hwc=False, augm=True):
    tfs = []
    if augm:
        tfs.append(RandomFlip(horizontal=True, vertical=True, hwc=hwc))
        tfs.append(RandomRotate(hwc=hwc))
    if crop_size is not None:
        # tfs.append(CenterCropPad(crop_size, wiggle=wiggle))
        tfs.append(RandomCropPad(crop_size))
    tfs.append(NumpyToTensor())
    train_transforms = transforms.Compose(tfs)
    print('train transforms:', train_transforms)
    return train_transforms

def _create_val_loaders(data_conf, compressed_dirs, fp16=True, num_workers=6,
                        pin_memory=False, val_split='validation', batch_size=1, **kwargs):
    val_transforms = _create_valid_transforms()
    # valid_ds = CompressedDataset(compressed_dirs, self.data_conf, split=val_split, flat=False, transform=val_transforms,
    #                              convert_f32=not fp16, **kwargs)
    eval_loaders = OrderedDict()
    splits = data_conf.get_splits()
    if val_split in splits:
        splits.remove(val_split)
        splits.insert(0, val_split)
    print('creating val loaders for %s' % str(splits))
    for split in splits:
        ds = CompressedDataset(compressed_dirs, data_conf, split=split, flat=False, transform=val_transforms,
                               convert_f32=not fp16, **kwargs)
        eval_loaders[split] = DataLoader(ds, num_workers=num_workers,  # collate_fn=collate_compressed_dict,
                                               pin_memory=pin_memory, batch_size=batch_size,
                                               persistent_workers=num_workers > 0)
        print('%s validation loader for %d samples in %d batches' % (split, len(ds), len(eval_loaders[split])))
    return eval_loaders

class CompressedDataModule(pl.LightningDataModule):
    def __init__(self, data_conf:DataConf, compressed_dirs, batch_size=1, crop_size=None,
                 num_workers=0, balance=True, pin_memory=False, augm_train=True,
                 fp16=False, val_split='validation', drop_last=False, **kwargs):
        super().__init__()
        # print('n train: %d, n valid: %d' % (self.n_train, self.n_valid))
        self.data_conf = data_conf
        self.compressed_dirs = compressed_dirs

        self.n_classes = 2
        self.batch_size = batch_size

        print('creating datasets')

        train_transforms = create_train_transforms(crop_size=crop_size, augm=augm_train)
        # train_transforms = create_train_transforms()
        train_ds = CompressedDataset(compressed_dirs, self.data_conf, split='training', flat=False, transform=train_transforms,
                                     convert_f32=not fp16, **kwargs)
        self.n_train = len(train_ds)


        sampler = None
        if balance:
            if str(balance).lower() not in ['true','1']:
                # weights = train_ds.balanced_category_weights()
                weights = train_ds.balanced_weights_col(balance)
            else:
                weights = train_ds.balanced_weights()
            sampler = WeightedRandomSampler(weights, len(weights))
            targets = train_ds.get_targets().argmax(axis=1)
            print('weighted sampler, first 20 weights:', to_numpy(weights[:20]))
            print('weighted sampler  first 20 targets:', targets[:20])
        else:
            print('training not balanced')
        print('creating train loader, num_workers=%d...' % num_workers, end=' ')
        self._train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, shuffle=sampler is None,
                                  num_workers=num_workers, #collate_fn=collate_compressed_dict,
                                  pin_memory=pin_memory,  persistent_workers=num_workers>0, drop_last=drop_last
                                  )
        print('%d train samples in %d batches' % (self.n_train, len(self._train_loader)))
        self._eval_loaders = _create_val_loaders(data_conf=data_conf, compressed_dirs=compressed_dirs, fp16=fp16,
                                                 num_workers=num_workers, pin_memory=pin_memory,
                                                 val_split=val_split, batch_size=1, #batch_size,
                                                 **kwargs)


    def get_eval_loaders(self):
        """ returns the loaders for evaluation (no augmentation), the one for validation first"""
        return self._eval_loaders

    def train_dataloader(self):
        return self._train_loader

    def val_dataloader(self):
        return self._eval_loaders['validation']

    def test_dataloader(self):
        return self.self._eval_loaders.get('testing',None)


class NicLearner(object):
    def __init__(self, net_conf, out_dir, enc_dim, data_config, preprocess_dir, batch_size=1,
                 train_type=TrainType.clf, pathes_csv=None, categories_order=None,
                 find_lr=False, monitor=None, monitor_mode='max', early_patience=25, crop_size=None,
                 epochs=150, precision=16, overwrite=False, overwrite_eval=False,
                 data_key='data', target_key='target', out_key='out', class_names=None,
                 anno_dir=None, n_hm_max=150, cache_dir=None, eval_out_dir=None,
                 autoname=True, exp_name='', copyconf=True, heatmap_purposes=['testing'],
                 no_heatmaps=False, no_out=False, hmsoft=False, seed=1, num_workers=6, pin_memory=True,
                 log_every_n_steps=20, inst_cluster=False, subtyping=False, group_weights=None,
                 opt_conf={'name':'adamw', 'lr':1e-4, 'weight_decay':1e-4}, lrs_conf={},
                 loss_conf={}, accelerator='gpu', # weight=None,
                 id_clf=False, balance=False, profiler=None, tpr_weight=None, **kwargs):
        """
        tpr_weight: relative weight of sensitivity to specificity when determining the clf-roc-threshold during evaluation
        """

        self._data_key = data_key
        self._target_key = target_key
        self._out_key = out_key

        self.net_conf = net_conf
        self.opt_conf = opt_conf
        if lrs_conf is None: lrs_conf = {}
        self.lrs_conf = lrs_conf
        if loss_conf is None: loss_conf = {}
        self.loss_conf = loss_conf
        self.enc_dim = enc_dim

        if monitor is None:
            monitor, monitor_mode = TrainType.monitor(train_type)
        if 'loss' in monitor:
            monitor_mode = 'min'

        self.data_conf = DataConf.from_config(data_config, train_type=train_type, target_names=class_names,
                                              categories_order=categories_order)
        if pathes_csv is not None:
            if not is_iterable(pathes_csv):
                pathes_csv = str(pathes_csv).split(',')
                pathes_csv = [pc.strip() for pc in pathes_csv]
            dfs = [pd.read_csv(pc, dtype={SlidesInfo.name_col:str}) for pc in pathes_csv]
            if len(dfs)==1:
                dfs = dfs[0]
            else:
                dfs = pd.concat(dfs, ignore_index=True)
            dfs.rename(columns={SlidesInfo.path_col:DataConf.image_col}, inplace=True)
            print('read %d pathes from %s ' % (len(dfs), str(pathes_csv)))
            if DataConf.image_col in self.data_conf.df:
                del self.data_conf.df[DataConf.image_col]
            self.data_conf.merge(dfs, SlidesInfo.name_col)
            print_df(self.data_conf.df.head(2))
        n_targets = self.data_conf.count_target_cols()
        self.train_type = train_type
        if train_type in [TrainType.clf,TrainType.multilabel]:
            if 'out_dim' in net_conf:
                if net_conf['out_dim']!=len(self.data_conf.target_cols):
                    raise ValueError('incompatible number of out dims %d and classes %d' % (net_conf['out_dim'], n_targets))
            net_conf['out_dim'] = n_targets
        if 'in_dim' in net_conf:
            if net_conf['in_dim']!=enc_dim:
                raise ValueError('network in_dim=%d != enc_dim=%d' % (net_conf['in_dim'], enc_dim))
        net_conf['in_dim'] = enc_dim
        if net_conf['name'] == 'dense':
            no_heatmaps = True

        self.tpr_weight = tpr_weight

        self.group_weights = group_weights

        self.compressed_dirs = preprocess_dir

        self.overwrite = overwrite
        self.overwrite_eval = overwrite_eval

        self.anno_dir = anno_dir
        self.n_hm_max = n_hm_max

        self.accelerator = accelerator
        self.batch_size = batch_size
        self.epochs = epochs
        self.precision = precision
        self.balance = balance
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.find_lr = find_lr
        self.early_patience = early_patience
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.log_every_n_steps = log_every_n_steps

        self.inst_cluster = inst_cluster
        if inst_cluster:
            self.net_conf['inst_cluster'] = True
        self.subtyping = subtyping

        self.kwargs = kwargs

        self.seed = seed
        self._time_info={}

        self.id_clf = id_clf

        if crop_size is not None and str(crop_size).lower()=='none':
            crop_size = None
        self.crop_size = crop_size

        #snellius specific
        scratch_node = '/scratch-node'
        if scratch_node in str(cache_dir):
            cdir = getpass.getuser()+'.'+os.getenv('SLURM_JOB_ID')
            if Path(scratch_node).exists():
                cache_root = scratch_node
            else:
                cache_root = '/scratch-local'
            cache_dir = cache_dir.replace(scratch_node, cache_root+'/'+cdir)
            print('using cache_dir %s' % cache_dir)
        self.cache_dir = cache_dir
        self.compressed_dirs_work = None

        self.no_out = no_out
        if no_out:
            no_heatmaps = True
        self.no_heatmaps = no_heatmaps
        if heatmap_purposes is None:
            heatmap_purposes = []
        elif is_string(heatmap_purposes):
            heatmap_purposes = heatmap_purposes.split(',')
        self.heatmap_purposes = heatmap_purposes
        self.hmsoft = hmsoft

        if autoname:
            exp_name = self._create_exp_name(exp_name)

        self.out_dir = Path(out_dir)
        if exp_name is not None and len(exp_name)>0:
            self.out_dir = self.out_dir/exp_name
        print('out_dir: %s' % str(self.out_dir))
        mkdir(self.out_dir)
        if copyconf:
            # shutil.copyfile(str(data_config), dst=str(self.out_dir/Path(data_config).name))
            self.data_conf.to_csv(str(self.out_dir/Path(data_config).name))

        self.eval_out_dir = self.out_dir
        if eval_out_dir is not None and eval_out_dir!=str(self.out_dir):
            self.eval_out_dir = eval_out_dir
            mkdir(eval_out_dir)

        self.dm = None

        if str(profiler)=='advanced':
            profiler = AdvancedProfiler(str(self.out_dir), 'profile')
        elif profiler:
            profiler = SimpleProfiler(str(self.out_dir), 'profile')
        self.profiler = profiler
        self.interrupted = False

    def _create_exp_name(self, exp_name_prefix):
        name = ''
        if exp_name_prefix is not None and len(exp_name_prefix)>0:
            name+=exp_name_prefix
            if not name.endswith('_'):
                name+='_'

        if is_iterable(self.compressed_dirs):
            cdir = self.compressed_dirs[0]
        else:
            cdir = self.compressed_dirs
        name += Path(cdir).name
        if self.early_patience != 25:
            name+='_ep%d' % self.early_patience
        name+='_fp16' if self.precision==16 else '_fp32'

        model = self._create_model(print_summary=False)
        parts = [name]
        parts.append(model.short_name)
        if not self.balance:
            parts.append('ub')
        elif str(self.balance).lower() not in ['true','1']:
            parts.append('%sb' % self.balance[:3])
        if self.crop_size is not None:
            parts.append('crop%d' % self.crop_size)
        if self.find_lr:
            parts.append('lrf')
        if 'adamw' not in self.opt_conf['name']:
            parts.append(self.opt_conf['name'])
        if 'plateau' not in self.lrs_conf.get('name','plateau'):
            parts.append(self.lrs_conf['name'])
        if 'bce' not in self.loss_conf.get('name','bce'):
            part = self.loss_conf['name']
            if 'cuts' in self.loss_conf:
                part+=str(len(self.loss_conf.get('cuts')))
            parts.append(part)
        if self.group_weights is not None and len(self.group_weights)>0:
            gw = ''.join([str(w) for w in self.group_weights])
            parts.append(gw)
        if self.loss_conf.get('weight',None) is not None:
            cw = self.loss_conf['weight']
            cws = [str(w).replace('0.','') for w in cw]
            parts.append('cw%s' % ''.join(cws))
        if self.loss_conf.get('label_smoothing',0)>0:
            lsmooth = int(self.loss_conf['label_smoothing']*100)
            parts.append('lsm%d' % lsmooth)
        if self.kwargs.get('swa',False):
            parts.append('swa')
        if 'accumulate_grad_batches' in self.kwargs:
            parts.append('agb%d' % self.kwargs['accumulate_grad_batches'])
        if self.kwargs.get('gradient_clip_val',0):
            parts.append(f'clip{self.kwargs["gradient_clip_val"]}')
        if self.id_clf:
            parts.append('id'+str(self.id_clf)[:3])
        if self.batch_size>1:
            parts.append(f'bs{self.batch_size}')
        parts.append(f'seed{self.seed}')
        if self.kwargs.get('overfit_batches',0):
            parts.append('ob')
        name = '_'.join(parts)
        name = name.replace('__','_')
        return name

    def _create_model(self, print_summary=True):
        model = create_nic_model(**self.net_conf)
        # if self.net_conf.get('act','relu')=='selu':
        #     model.apply(WeightsInit(self.net_conf.get('act','linear')))
        # else:
        #     model.apply(weights_init)
        # model.apply(WeightsInit())
        model.apply(weights_init)

        size =[128, 128] if self.crop_size is None else [self.crop_size, self.crop_size]
        if print_summary:
            inp = torch.zeros((1,self.enc_dim,size[0],size[1]))
            inp[0,0,0,0] = 1
            print_model_summary(model, inp, out_path=self.out_dir/'model_summary.txt')
        return model

    def _cache_compressed(self):
        if self.compressed_dirs_work is not None:
            return #already cached

        if self.cache_dir is None:
            self.compressed_dirs_work = self.compressed_dirs
        else:
            self.compressed_dirs_work = cache_dirs(self.compressed_dirs, self.cache_dir, name_depth=4)
        print('compressed_dirs:', self.compressed_dirs_work)

    def train(self):
        # set_seed(self.seed)
        # torch.manual_seed(self.seed)
        seed_everything(self.seed, workers=True)

        self._cache_compressed()
        self.dm = CompressedDataModule(data_conf=self.data_conf, compressed_dirs=self.compressed_dirs_work,
                                       batch_size=self.batch_size, crop_size=self.crop_size,
                                       fp16=self.precision==16, pin_memory=self.pin_memory,
                                       num_workers=self.num_workers, balance=self.balance, drop_last=self.data_conf.is_surv())

        finished_path = self.out_dir/'training_finished.txt'
        if finished_path.exists():
            print('training finished')
            return

        resume_path = self.out_dir/'last.ckpt'
        if resume_path.exists():
            print('found resume path %s' % str(resume_path))
            resume_path = str(resume_path)
            if self.find_lr:
                lr = self._get_found_lr()
                # print('resuming with found lr %.4e' % lr)
                #this shouldn't have any effect since restored from ckpt
                self.opt_conf['lr'] = lr
                self.find_lr = False

        else:
            resume_path = None

        train_timer = Timer('nic train')
        # Samples required by the custom ImagePredictionLogger callback to log image predictions.
        # val_samples = next(iter(dm.val_dataloader()))
        # val_imgs, val_labels = val_samples[0], val_samples[1]
        # val_imgs.shape, val_labels.shape

        model = self._create_model()

        if self.lrs_conf.get('name','plateau')=='plateau':
            self.lrs_conf.update(dict(monitor=self.monitor, mode=self.monitor_mode))
        module_params = dict(model=model, opt=self.opt_conf, lrs=self.lrs_conf, loss = self.loss_conf,
                             data_key=self._data_key, target_key=self._target_key)

        # if self.net_conf.get('relu_loss',0):
        #     module_params['additional_out_keys'] = ['relu_out']

        module_params['additional_losses'] = []
        if self.inst_cluster:
            loss_adapter = InstClusterLoss(n_classes=self.net_conf['out_dim'], subtyping=self.subtyping)
            module_params.get('additional_losses').append(loss_adapter)

        if self.group_weights:
            loss_adapter = ClfGroupLoss(self.group_weights)
            module_params.get('additional_losses').append(loss_adapter)
            module_params['average_metrics'] = {f'{ph}_auc_avg':[f'{ph}_auc', f'{ph}_group_auc'] for ph in ['val','train','test']}

        if self.train_type==TrainType.regr:
            module = RegrModule(**module_params)
        elif self.train_type==TrainType.surv:
            from wsilearn.dl.pl.pl_modules_surv import SurvModule
            module = SurvModule(**module_params)
        else:
            module = ClfModule(n_classes=len(self.data_conf.target_cols),
                               multilabel=self.data_conf.is_multilabel(),
                               **module_params)

        # Initialize wandb logger
        # wandb_logger = WandbLogger(project=NIC, job_type='train', id=exp_name,
        #                            name=exp_name, offline=wandb_offline)
        csv_logger = CSVLogger(str(self.out_dir), name='logs')
        mylogger = EpochCsvLogger(self.out_dir, epoch_col='Epoch')
        plot_cb = HistoryPlotCallback(mylogger.out_path, x_col='Epoch')
        mylogger.final_fct = plot_cb
        # tb_logger = pl_loggers.TensorBoardLogger(str(self.out_dir), name='tensorboard')
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        print(f'early stopping on {self.monitor}, mode {self.monitor_mode}, patience {self.early_patience}')
        early_stop_callback = EarlyStopping(monitor=self.monitor, mode=self.monitor_mode, patience=self.early_patience,
                                            strict=False, verbose=False)#strict=True causes an error on resume

        Path(self.out_dir).mkdir(exist_ok=True, parents=True)

        checkpoint_callback = ModelCheckpoint(monitor=self.monitor, mode=self.monitor_mode, save_last=True, save_top_k=1,
                                              dirpath=str(self.out_dir), filename='best_{epoch}_{%s:.2f}' % self.monitor,
                                              )

        # Initialize a trainer
        additional_kwargs = {}
        print('training with precision', self.precision)
        # additional_kwargs['plugins'] = [SLURMEnvironment(auto_requeue=False)]
        # print('adding slurm auto_requeue=False plugin')
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor, plot_cb,
                   # SaveValidationOutput(self.out_dir)
                   ]
        swa = self.kwargs.pop('swa',False)
        if swa:
            start_epoch = self.kwargs.pop('swa_epoch_start',10)
            print('adding swa callback starting at epoch %d')
            callbacks.append(StochasticWeightAveraging(swa_epoch_start=start_epoch,
                                                       swa_lrs=None, annealing_epochs=self.kwargs.pop('swa_annealing_epochs',3),
                                                       device=None)) #swa_lrs=1e-2
        print('trainer kwargs:', self.kwargs)
        callbacks.append(MeterlessProgressBar(refresh_rate=10))
        trainer = pl.Trainer(max_epochs=self.epochs,
                             # progress_bar_refresh_rate=10,
                             accelerator=self.accelerator, devices=1, benchmark=self.batch_size>1,
                             logger=[csv_logger, mylogger
                                     # wandb_logger, tb_logger,
                                     ],
                             log_every_n_steps=self.log_every_n_steps,
                             num_sanity_val_steps=0,
                             callbacks=callbacks,
                             precision=self.precision,
                             enable_progress_bar=True,
                             default_root_dir=str(self.out_dir),
                             profiler=self.profiler,
                             # checkpoint_callback=checkpoint_callback,
                             # resume_from_checkpoint=resume_path
                             # auto_lr_find=True,
                             **additional_kwargs, **self.kwargs
                             )
        self._find_lr(trainer, module) #not used

        trainer.fit(module, datamodule=self.dm, ckpt_path=resume_path)
        train_time = train_timer.stop()
        epoch_time = train_time/(trainer.current_epoch+1)
        self._time_info.update({'train_time':train_time, 'train_time_str':time_to_str(train_time),
                                'train_time_per_epoch':epoch_time, 'train_time_per_epoch_str':time_to_str(epoch_time)})
        self.status = trainer.state.status
        if self.status == TrainerStatus.INTERRUPTED:
            self.interrupted = True
        else:
            print('Done training!')
            finished_path.touch(exist_ok=True)

        # create_tensorboard_start_text(Path(self.out_dir)/'logs/version_0')

        # print('test...')
        # trainer.test(module, datamodule=dm, ckpt_path=ckpt_path)
        # wandb.finish()

    def _find_lr(self, trainer, module):
        """ lr_auto_find from Trainer doesnt work for some reason (maybe because of early stopping) """
        if not self.find_lr:
            return
        # trainer.tune(module)

        lr_finder = trainer.tuner.lr_find(module, datamodule=self.dm)
        # print(lr_finder.results)
        new_lr = lr_finder.suggestion()
        print('lr:', new_lr)
        fig = lr_finder.plot(suggest=True)
        # if is_me():
        #     plt.show(block=True)
        lr_find_out_path = self.out_dir/f'find_lr_{new_lr:.4e}.jpg'
        plt.savefig(str(lr_find_out_path), bbox_inches='tight')
        module.set_lr(new_lr)
        # # Pick point based on plot, or get suggestion
        # # update hparams of the model
        # model.hparams.lr = new_lr

    def _get_found_lr(self):
        lr_path = PathUtils.list_pathes(self.out_dir, containing='find_lr')[0]
        lr = float(lr_path.stem.split('_')[-1])
        return lr

    def _find_best_model_path(self):
        return find_pl_model_path(self.out_dir)

    def _get_last_model_path(self):
        return find_pl_model_path(self.out_dir, last=True)

    def _get_eval_loaders(self, **kwargs):
        if self.dm is None:
            return _create_val_loaders(data_conf=self.data_conf, compressed_dirs=self.compressed_dirs,
                    fp16=self.precision==16, num_workers=self.num_workers, pin_memory=self.pin_memory, **kwargs)

        else:
            return self.dm.get_eval_loaders()

    def eval_config(self, last=False, clf_thresholds=None):
        if last:
            out_dir = Path(self.eval_out_dir)/'eval_last'
            mkdir(out_dir)
            model_path = self._get_last_model_path()
        else:
            out_dir = Path(self.eval_out_dir)
            model_path = self._find_best_model_path()

        clf_results_out = out_dir/ 'results.json'
        if clf_results_out.exists() and not self.overwrite_eval:
            print('skipping evaluation, due to already existing %s' % str(clf_results_out))
            eval_results = read_json_dict(clf_results_out)
            print(eval_results)
            return


        timer = Timer('nic eval')
        model = self._create_model()
        #old version compatibility
        load_pl_state_dict(model=model, path=model_path, replacements={'att_net.attention_a':'att_net.att_m',
                                                                             'att_net.attention_b':'att_net.att_gate',
                                                                             'att_net.attention_c':'att_net.att_last'})

        inf_out_writer = InferenceOutWriter(overwrite=self.overwrite_eval, out_dir_name='out_npz')
        inferencer = Inferencer(model, post_fct=TrainType.post_fct(self.train_type), overwrite=self.overwrite_eval,
                                callbacks=[] if self.no_out else [inf_out_writer])
        hm_writer = HeatmapWriter(npz_dir=None, anno_dir=self.anno_dir, #hmsoft=self.hmsoft,
                                  n_hm_correct_max=self.n_hm_max, n_hm_wrong_max=self.n_hm_max,
                                  overwrite=self.overwrite_eval, train_type=self.train_type)
        callbacks=[] if self.no_heatmaps else [hm_writer]
        if self.train_type in [TrainType.clf,TrainType.multilabel]:
            hm_writer = ClfEvaluationHeatmapWriter(npz_dir=None, anno_dir=self.anno_dir, #hmsoft=self.hmsoft,
                                                   n_hm_correct_max=self.n_hm_max, n_hm_wrong_max=self.n_hm_max,
                                                   overwrite=self.overwrite_eval)
            callbacks=[] if self.no_heatmaps else [hm_writer]
            evaluator = ClfEvaluator(class_names=self.data_conf.target_cols, callbacks=callbacks,
                                     tpr_weight=self.tpr_weight, clf_thresholds=clf_thresholds)
        elif self.train_type == TrainType.surv:
            evaluator = SurvEvaluator(callbacks=callbacks)
        elif self.train_type == TrainType.regr:
            evaluator = RegrEvaluator(callbacks=callbacks)
        else:
            raise ValueError('implement non clf')

        results = {}
        for purp, loader in self._get_eval_loaders().items():
            eval_out_dir = out_dir/('eval_%s' % purp)
            print('%s eval out_dir: %s' % (purp, str(eval_out_dir)))
            disable_hm = purp not in self.heatmap_purposes and 'all' not in self.heatmap_purposes
            hm_writer.disabled=disable_hm
            inf_out_writer.disabled=disable_hm
            if not disable_hm:
                hm_writer.npz_dir = eval_out_dir/inf_out_writer.out_dir_name
            dfp = inferencer.apply(loader, eval_out_dir)
            # data_conf = DataConf(dfp, train_type=self.train_type, target_names=self.data_conf.target_cols)
            data_conf = DataConf.like(dfp, self.data_conf)
            data_conf.to_csv(eval_out_dir/'data_conf.csv')
            pmetrics = evaluator.evaluate_all(data_conf, out_dir=eval_out_dir)
            results[purp] = pmetrics
            # create_short_info_file(eval_out_dir, results, purposes={purp:purp})

        self._time_info.update(dict(inf_out_writer=inf_out_writer.print_time(), hm_writer=hm_writer.print_time(),
                eval_time=time_to_str(timer.stop())))
        other_results_path = Path(out_dir)/'rest_results.json'
        other_results = dict(time=self._time_info)
        write_json_dict(path=other_results_path, data=other_results)

        results['time'] = self._time_info
        try:
            write_json_dict(path=clf_results_out, data=results)
        except Exception as ex:
            print('writing results failed maybe because of time info: %s' % str(ex))
            del results['time']
            write_json_dict(path=clf_results_out, data=results)
        short_info_str = create_short_info_file(out_dir, results)
        print('Done %s' % short_info_str)




def train_nic():
    # print('CUDA_VISIBLE_DEVICES:',os.environ.get('CUDA_VISIBLE_DEVICES',''))
    if torch.cuda.is_available():
        print('torch cuda available')
    else:
        raise ValueError('cuda not available!')

    parser = FlexArgumentParser()

    # Add arguments to the parser
    # parser.add_argument('--net_conf', type=str, help='network config')
    parser.add_argument('--out_dir', type=str, help='output directory', required=True)
    parser.add_argument('--enc_dim', type=int, help='encoding dimension', required=True)
    parser.add_argument('--data_config', type=str, help='csv configuration', required=True)
    parser.add_argument('--preprocess_dir', type=str, help='compressed file directories, separated by comma')
    parser.add_argument('--batch_size', type=int, default=1, help='specify crop size if > 1', required=False)
    parser.add_argument('--train_type', type=str, default='clf', choices=('clf,multilabel,surv,regr'))
    parser.add_argument('--pathes_csv', type=str, help='csv with name,path', required=False)
    # parser.add_argument('--categories_order', type=str, help='Description of categories_order parameter')
    # parser.add_argument('--find_lr', action='store_true', help='Description of find_lr parameter')
    parser.add_argument('--monitor', type=str, help='e.g. val_auc', required=False, default='val_auc')
    parser.add_argument('--monitor_mode', type=str, help='max or min', required=False, default='max')
    parser.add_argument('--early_patience', type=int, required=False, default=25)
    parser.add_argument('--crop_size', type=str, required=False)
    parser.add_argument('--epochs', type=int, default=150, required=False)
    parser.add_argument('--precision', type=int, default=16, choices=(16,32), required=False)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--overwrite_eval', action='store_true')
    # parser.add_argument('--data_key', type=str, help='Description of data_key parameter')
    # parser.add_argument('--target_key', type=str, help='Description of target_key parameter')
    # parser.add_argument('--out_key', type=str, help='Description of out_key parameter')
    parser.add_argument('--class_names', type=str, help='hot-encoded class names in the config', required=True)
    parser.add_argument('--anno_dir', type=str, help='annotation directory for visualizations', required=False)
    parser.add_argument('--n_hm_max', type=int, default=150, help='maximal heatmaps', required=False)
    parser.add_argument('--cache_dir', type=str, required=False)
    parser.add_argument('--eval_out_dir', type=str, required=False)
    parser.add_argument('--autoname', action='store_true', help='auto-create experiment name/directory')
    parser.add_argument('--exp_name', type=str, help='manual experiment name', required=False)
    # parser.add_argument('--copyconf', action='store_true', help='Description of copyconf parameter')
    # parser.add_argument('--heatmap_purposes', nargs='+', default=['testing'], help='Description of heatmap_purposes parameter')
    parser.add_argument('--no_heatmaps', action='store_true', help='dont save heatmaps')
    parser.add_argument('--no_out', action='store_true', help='dont save out npz per slide', required=False)
    # parser.add_argument('--hmsoft', action='store_true', help='Description of hmsoft parameter')
    parser.add_argument('--seed', type=int, default=1, help='set seed', required=False)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--pin_memory', action='store_true')
    # parser.add_argument('--log_every_n_steps', type=int, default=20, help='Description of log_every_n_steps parameter')
    parser.add_argument('--inst_cluster', action='store_true', help='clams instance clustering')
    parser.add_argument('--subtyping', action='store_true', help='clams subtyping')
    parser.add_argument('--group_weights', type=str, help='second cse loss weighting, e.g. [0,1,1] '
                                                          'in order of the class_names')
    # parser.add_argument('--opt_conf', type=str, help='optimizer config, e.g. {"name":"adamw", "lr":3e-4, "weight_decay":1e-4}', required=False)
    # parser.add_argument('--lrs_conf', type=str, help='learning rate scheduler config, e.g. {"name":"cyclic"} or {"patience":2}', required=False)
    # parser.add_argument('--loss_conf', type=str, help='Description of loss_conf parameter')
    parser.add_argument('--balance', help='if True balances the classes, if same as category column, '
                                          'uses the category values', action='store_true')
    # parser.add_argument('--profiler', type=str, help='Description of profiler parameter')
    # parser.add_argument('--tpr_weight', type=str, help='gives heigher weight to positives for binary evaluation', required=False)

    args = parser.parse_args()
    _train_nic(args)

def _train_nic(args):
    print('nic args:', args)
    eval_last = args.pop('eval_last',False)
    if 'out_dir' not in args:
        args['out_dir'] = args.pop('output_dir')
    trainer = NicLearner(**args)
    print('out dir: %s' % str(trainer.out_dir))
    write_json_dict(path=trainer.out_dir/'args.json', data=args)
    trainer.train()
    if not trainer.interrupted:
        if eval_last:
            trainer.eval_config(last=True)
        trainer.eval_config(last=False)


def main():
    train_nic()

if __name__ == '__main__':
    main()