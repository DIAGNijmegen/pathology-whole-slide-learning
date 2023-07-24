from pathlib import Path

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, StochasticWeightAveraging, EarlyStopping, \
    ModelSummary
from pytorch_lightning.loggers import CSVLogger
# from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.trainer.states import TrainerStatus

from wsilearn.utils.cool_utils import mkdir, is_iterable, Timer, time_to_str, create_tensorboard_start_text, read_json_dict, \
    write_json_dict
from wsilearn.dl.pl.pl_logger import EpochCsvLogger, HistoryPlotCallback, MeterlessProgressBar
from pytorch_lightning import loggers as pl_loggers, seed_everything, LightningDataModule

from wsilearn.dl.pl.pl_inference import find_pl_model_path, load_pl_state_dict, InferenceOutWriter, Inferencer
from wsilearn.dl.torch_utils import print_model_summary
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from wsilearn.utils.path_utils import PathUtils

class PlTrainer(object):
    def __init__(self, module, out_dir, in_dim, in_size, batch_size=1,
                 find_lr=False, monitor='val_auc', monitor_mode='max', early_patience=25,
                 epochs=200, precision=16, overwrite=False, overwrite_eval=False,
                 data_key='data', target_key='target', out_key='out',
                 cache_dir=None, exp_name=None, exp_name_prefix='',
                 seed=1, num_workers=8, log_every_n_steps=50, gpus=1,
                 opt={'name': 'adamw', 'lr':1e-4, 'weight_decay':1e-4},
                 loss={}, model_summary_shape=None,
                 profiler=None, save_every_n_epochs=None, **kwargs):

        self._data_key = data_key
        self._target_key = target_key
        self._out_key = out_key

        self.module = module
        self.opt_conf = opt
        self.loss_conf = loss
        self.out_dir = Path(out_dir)
        self.overwrite = overwrite
        self.overwrite_eval = overwrite_eval

        self.batch_size = batch_size
        self.epochs = epochs
        self.precision = precision
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.find_lr = find_lr
        self.early_patience = early_patience
        self.num_workers = num_workers
        self.log_every_n_steps = log_every_n_steps
        self.gpus = gpus
        self.kwargs = kwargs

        self.seed = seed
        self.profiler = profiler
        self._time_info={}

        self.cache_dir = cache_dir

        self.save_every_n_epochs = save_every_n_epochs

        self.in_size = in_size
        if exp_name is None:
            exp_name = self._create_exp_name(exp_name_prefix)
        if len(exp_name)>0:
            self.out_dir = self.out_dir/exp_name

        mkdir(self.out_dir)
        self.interrupted = False

        print_model_summary(module, (in_dim, in_size, in_size), out_path=self.out_dir/'model_summary.txt')


    def _create_exp_name(self, exp_name_prefix):
        parts = []
        if exp_name_prefix is not None and len(exp_name_prefix)>0:
            parts.append(exp_name_prefix)
        if hasattr(self.module, 'short_name'):
            parts.append(self.module.short_name)
        if self.early_patience != 25:
            parts.append('ep%d')
        if self.precision !=16:
            parts.append(f'fp{self.precision}')
        parts.append('in%d' % self.in_size)
        # if self.find_lr:
        #     parts.append('lrf')
        if 'adam' not in self.opt_conf['name']:
            parts.append(self.opt_conf['name'])
        if self.loss_conf.get('weight',None) is not None:
            cw = self.loss_conf['weight']
            cws = [str(w).replace('0.','') for w in cw]
            parts.append('cw%s' % ''.join(cws))
        if self.kwargs.get('swa',False):
            parts.append('swa')
        if 'accumulate_grad_batches' in self.kwargs:
            parts.append('agb%d' % self.kwargs['accumulate_grad_batches'])
        if self.batch_size>1:
            parts.append(f'bs{self.batch_size}')
        parts.append(f'seed{self.seed}')

        name = '_'.join(parts)
        name = name.replace('__','_')
        return name


    def train(self, dm):
        # set_seed(self.seed)
        # torch.manual_seed(self.seed)
        seed_everything(self.seed, workers=True)

        self.dm = dm

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
                self.opt_conf['lr'] = lr
                self.find_lr = False

        else:
            resume_path = None

        train_timer = Timer('train')


        # Initialize wandb logger
        # wandb_logger = WandbLogger(project='pl_cifar10mini', job_type='train', id=exp_name,
        #                            name=exp_name, offline=wandb_offline)
        csv_logger = CSVLogger(str(self.out_dir), name='logs')
        mylogger = EpochCsvLogger(self.out_dir, epoch_col='Epoch')
        plot_cb = HistoryPlotCallback(mylogger.out_path, x_col='Epoch')
        mylogger.final_fct = plot_cb
        # tb_logger = pl_loggers.TensorBoardLogger(str(self.out_dir), name='tensorboard')
        # lr_monitor = LearningRateMonitor(logging_interval='epoch')
        lr_monitor = LearningRateMonitor(logging_interval='epoch')

        early_stop_callback = EarlyStopping(monitor=self.monitor, mode=self.monitor_mode, patience=self.early_patience,
                                            strict=False, verbose=False)#strict=True causes an error on resume

        Path(self.out_dir).mkdir(exist_ok=True, parents=True)

        checkpoint_callback = ModelCheckpoint(monitor=self.monitor, mode=self.monitor_mode, save_last=True, save_top_k=1,
                                              dirpath=str(self.out_dir), filename='best_{epoch}_{%s:.2f}' % self.monitor,
                                              every_n_epochs=self.save_every_n_epochs)

        # Initialize a trainer
        additional_kwargs = {}
        print('training with precision %d' % self.precision)
        # if not is_me():
        #     additional_kwargs['plugins'] = [SLURMEnvironment(auto_requeue=False)]
        #     print('adding slurm auto_requeue=False plugin')
        # model_summary_cb = ModelSummary(max_depth=4) #doesnt show shape

        ### callbacks
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor, #plot_cb
                   # ImagePredictionLogger(val_samples)
                   ]
        swa = self.kwargs.pop('swa',False)
        if swa:
            start_epoch = self.kwargs.get('swa_epoch_start',10)
            print('adding swa callback starting at epoch %d')
            callbacks.append(StochasticWeightAveraging(swa_epoch_start=start_epoch,
                                                       swa_lrs=None, annealing_epochs=self.kwargs.get('swa_annealing_epochs',5),
                                                       device=None)) #swa_lrs=1e-2
        # callbacks.append(MeterlessProgressBar(refresh_rate=5))

        trainer = pl.Trainer(max_epochs=self.epochs,
                             # progress_bar_refresh_rate=10,
                             gpus=self.gpus, benchmark=self.batch_size>1,
                             logger=[csv_logger, mylogger,
                                     # wandb_logger,
                                     ],
                             log_every_n_steps=self.log_every_n_steps,
                             num_sanity_val_steps=0,
                             callbacks=callbacks,
                             precision=self.precision,
                             enable_progress_bar=True,
                             profiler=self.profiler,
                             **additional_kwargs, **self.kwargs
                             )
        self._find_lr(trainer)

        trainer.fit(self.module, datamodule=self.dm, ckpt_path=resume_path)

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

    def eval_dm(self, dm:LightningDataModule, **kwargs):
        model = self.module.model
        model_path = find_pl_model_path(self.out_dir)
        load_pl_state_dict(model=model, path=model_path)
        inferencer = Inferencer(model, overwrite=self.overwrite_eval,
                                callbacks=[], **kwargs)
        for (phase, loader) in (('val', dm.val_dataloader()),('test',dm.test_dataloader())):
            if loader is not None:
                phase_out_dir = self.out_dir/('eval_'+phase)
                inferencer.apply(loader, phase_out_dir)

    def _find_lr(self, trainer):
        """ lr_auto_find from Trainer doesnt work for some reason (maybe because of early stopping) """
        if not self.find_lr:
            return
        # trainer.tune(module)
        module = self.module

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
        # return Path(self.out_dir)/'last.ckpt'

    def eval(self, last=False):
        if last:
            out_dir = Path(self.out_dir)/'eval_last'
            mkdir(out_dir)
            model_path = self._get_last_model_path()
        else:
            out_dir = Path(self.out_dir)
            model_path = self._find_best_model_path()

        results_out = out_dir/ 'results.json'
        if results_out.exists() and not self.overwrite_eval:
            print('skipping evaluation, due to already existing %s' % str(results_out))
            eval_results = read_json_dict(results_out)
            print(eval_results)
            return


        timer = Timer('nic eval')
        load_pl_state_dict(model=self.module, path=model_path)


        results = self._eval()

        results['eval_time'] = time_to_str(timer.stop())
        try:
            write_json_dict(path=results_out, data=results)
        except Exception as ex:
            print('writing results failed maybe because of time info: %s' % str(ex))
            del results['time']
            write_json_dict(path=results_out, data=results)
        # short_info_str = create_short_info_file(out_dir, results)
        # print('Done %s' % short_info_str)
