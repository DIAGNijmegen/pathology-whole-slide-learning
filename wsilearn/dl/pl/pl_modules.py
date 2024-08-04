# lightning related imports
from collections import defaultdict
from copy import deepcopy, copy

import torch, sys
import pytorch_lightning as pl
import torchmetrics
# from pytorch_lightning.metrics.functional import accuracy

from wsilearn.utils.cool_utils import is_dict, is_string, is_iterable, is_list
from wsilearn.dl.loss_utils import create_loss_fct
from wsilearn.dl.models.model_utils import create_optimizer, create_lr_scheduler
from wsilearn.dl.torch_utils import to_cpu, to_numpy
from torch.nn import functional as F, Linear, ModuleList
import numpy as np
import torch.nn as nn

lrs_plateau = {'name':'plateau', 'monitor': 'val_loss', 'mode': 'min', 'factor':0.1, 'min_lr':1e-7, 'patience':10}

class ModuleBase(pl.LightningModule):
    def __init__(self, model, loss,
                 opt={'name': 'adamw', 'lr':1e-4, 'weight_decay':1e-4},
                 lrs=lrs_plateau,
                 data_key='data', target_key='target', out_key='out',
                 n_features=None, n_train=None,
                 epoch_metrics=[], average_metrics={},
                 additional_out_keys=[], additional_losses=[], phases=['train','val','test'],
                 ):
        """
        average_metrics: e.g. {'auc_avg':['auc1','auc2']} - logs metric averages, e.g.
        additional_out_keys: multi-task loss using same loss fct and target

         """
        super().__init__()
        self._data_key = data_key
        self._target_key = target_key
        self._out_key = out_key

        self._epoch_metrics = epoch_metrics
        self._average_metrics = average_metrics

        self._phases = phases
        self.metrics = {}
        for phase in self._phases:
            self._add_metrics(phase)

        self.additional_out_keys = additional_out_keys

        if additional_losses is None:
            additional_losses = []
        else:
            if not is_iterable(additional_losses):
                additional_losses = [additional_losses]
            for lm in additional_losses:
                lm.set_module(self)
                for phase in self._phases:
                    self.metrics.update(lm.get_metrics(phase))

        self.additional_losses = additional_losses

        hyper_params = {}
        hyper_params['opt'] = opt.copy()
        if is_dict(loss):
            hyper_params['loss'] = loss.copy()
            self.criterion = create_loss_fct(**loss)
        else:
            self.criterion = loss

        if lrs is None:
            lrs = {}
        if 'name' not in lrs:
            lrs_def = copy(lrs_plateau)
            lrs_def.update(lrs)
            lrs = lrs_def
        hyper_params['lrs'] = lrs
        self.save_hyperparameters(hyper_params, ignore=['model'])
        print('hyperparams:', hyper_params)

        self.model = model

        # self.opt_params = opt_params

        self._last_losses = defaultdict(list)
        self.n_train = n_train


    def forward(self, x, **kwargs):
        if is_dict(x):
            x = x[self._data_key]
        if x.isnan().any():
            print('x has nans!')
        out = self.model(x, **kwargs)
        return out

    def _compute_metrics_step(self, phase, target, logits, prog_bar=True, **log_kwargs):
        # # this will reset the metric automatically at the epoch end
        # self.train_acc(preds, y)
        # self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)
        if phase not in self.metrics:
            return
        metrics = self.metrics[phase]
        target, logits = to_cpu(target, logits)
        pred, target = self._post_process(logits, target)
        for name, metric in metrics.items():
            log_name = phase+"_"+name+'_step'
            if name in self._epoch_metrics:
                metric.update(pred, target)
            else:
                val = metric(pred, target)
                self.log(log_name, val, prog_bar=prog_bar, on_step=True, on_epoch=False, batch_size=len(pred), **log_kwargs)

    def _compute_metrics_epoch(self, phase, **log_kwargs):
        if phase not in self.metrics:
            return
        metrics = self.metrics[phase]


        for name, metric in metrics.items():
            debug_kwargs = {}
            # if name=='cind':
            #     debug_kwargs.update(dict(epoch=self.trainer.current_epoch, phase=phase, out_dir=self.trainer.log_dir))

            log_name = phase+"_"+name
            try:
                val = metric.compute(**debug_kwargs)
            except:
                print('ERROR IN METRIC %s!' % log_name)
                print(sys.exc_info())
                val = -1
            self.epoch_metrics[log_name] = to_numpy(val) if torch.is_tensor(val) else val
            self.log(log_name, val, on_step=False, on_epoch=True, **log_kwargs)
            metric.reset()


    def _aggregate_loss(self, loss, batch_idx, phase):
        if loss is not None:
            self.log(f"{phase}_loss_step", loss, batch_size=self._batch_size)
            if batch_idx==0:
                self._last_losses[phase] = []
            self._last_losses[phase].append(to_numpy(loss))
        else:
            print(f'{phase} loss is None!')

    def _compute_overall_loss(self, result, y, loss, phase):
        for okey in self.additional_out_keys:
            logitsi = self._get_logits(result, out_key=okey)
            lossi = self._compute_loss(logitsi, y, phase=phase)
            loss = loss + lossi
        if len(self.additional_out_keys):
            loss = loss / (1+len(self.additional_out_keys))
        return loss

    def training_step(self, batch, batch_idx):
        logits, loss = self._step(batch, batch_idx, 'train')
        return loss

    def _step(self, batch, batch_idx, phase):
        x = batch[self._data_key]
        y = self._get_targets(batch)
        self._batch_size = len(x)
        result = self(x)
        logits = self._get_logits(result)
        loss = self._compute_loss(logits, y, phase=phase)
        loss = self._compute_overall_loss(result, y, loss, phase=phase)
        self._compute_metrics_step(target=y, logits=logits, phase=phase)
        self._aggregate_loss(loss, batch_idx, phase=phase)
        if loss is not None:
            for la in self.additional_losses:
                loss = la(batch, batch_idx, loss, result, x, y, phase=phase)
        return logits, loss

    def test_step(self, batch, batch_idx):
        logits, loss = self._step(batch, batch_idx, 'test')
        return logits

    def validation_step(self, batch, batch_idx):
        logits, loss = self._step(batch, batch_idx, 'val')
        return logits

    def _compute_loss(self, logits, y, phase):
        return self.criterion(logits, y)

    def _get_logits(self, result, out_key=None):
        if out_key is None:
            out_key = self._out_key
        if is_dict(result):
            # result = result.pop(out_key)
            result = result.get(out_key)
        # self._batch_size = len(result)
        return result

    def _get_targets(self, batch):
        return batch[self._target_key]

    def _post_process(self, logits, target):
        return logits, target


    def _log_loss_epoch(self, phase):
        losses = self._last_losses[phase]
        mean_loss = np.mean(losses)
        self.log('Epoch', float(self.trainer.current_epoch), on_step=False, on_epoch=True)
        self.log('%s_loss' % phase, mean_loss, on_step=False, on_epoch=True)
        #on_step=True raises error since on epoch_end
        self._last_losses[phase] = []

    def _epoch_end(self, outputs, phase):
        phase_metrics = [phase]
        for lmd in self.additional_losses:
            phase_metrics.extend(lmd.get_metric_names(phase))

        self.epoch_metrics = {}
        for phase_name in phase_metrics:
            self._log_loss_epoch(phase=phase_name)
            self._compute_metrics_epoch(phase=phase_name)

        if self._average_metrics is not None and len(self._average_metrics)>0:
            for avname, mnames in self._average_metrics.items():
                vals = [self.epoch_metrics.get(mname,None) for mname in mnames]
                if None in vals:
                    continue
                avg = np.mean(vals)
                self.log(avname, avg, on_step=False, on_epoch=True)

    def training_epoch_end(self, outputs):
        return self._epoch_end(outputs, 'train')

    def validation_epoch_end(self, outputs):
        return self._epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs):
        return self._epoch_end(outputs, 'test')


    def configure_optimizers(self):
        optimizer = create_optimizer(model_parameters=self.parameters(), **self.hparams['opt'])
        lrs_params = deepcopy(self.hparams['lrs'])
        monitor = lrs_params.pop('monitor', 'val_loss')
        # sname = lrs_params.pop('name','plateau')
        lr = self.hparams['opt']['lr']
        lrs_pl_kwargs = {}
        if lrs_params['name'] == 'cyclic':
            if 'base_lr' not in lrs_params:
                lrs_params['base_lr']= lr/10
            if 'max_lr' not in lrs_params:
                lrs_params['max_lr']= lr*10
            if 'adam' in self.hparams['opt']['name']:
                lrs_params['cycle_momentum'] = False
        else:
            lrs_pl_kwargs.update(dict(monitor=monitor, interval='epoch', frequency=1))

        scheduler = create_lr_scheduler(optimizer=optimizer, **lrs_params)
        print('created optimizer and scheduler with hparams', str(self.hparams))
        # return {'optimizer':optimizer, 'lr_scheduler':scheduler,
        #         'monitor':self.hparams['lrs']['monitor']}
        return {'optimizer':optimizer,
                'lr_scheduler':dict(scheduler=scheduler, strict=False,  **lrs_pl_kwargs),
                }

    def set_lr(self, lr):
        self.hparams['opt']['lr'] = lr


class ClfModule(ModuleBase):
    def __init__(self, n_classes, loss=None, multilabel=False, epoch_metrics=['auc'], **kwargs):
        """ loss: name of the loss, dictionary(loss=Name,etc) or None """
        self.n_classes = n_classes
        self.multilabel = multilabel

        if multilabel:
            loss_params = {'name':'bcelogits'}
        else:
            loss_params = {'name':'crossentropy'}

        if loss is not None:
            if is_string(loss):
                loss_params['name'] = loss
            elif is_dict(loss):
                loss_params.update(loss)

        super().__init__(loss=loss_params, epoch_metrics=epoch_metrics, **kwargs)


    def _post_process(self, logits, target):
        if target is not None and target.dtype==torch.float: #bcelogits expects float, but Accuracy int
            target = target.to(torch.int)
        logits = logits.to(torch.float32)
        if self.multilabel:
            probs = torch.sigmoid(logits)
        else:
            probs = logits.softmax(dim=-1)
        return probs, target

    def _add_metrics(self, phase):
        metrics = self.metrics
        n_classes = self.n_classes
        # if torchmetrics.__version__ < '1.0.0':
        #     print('torchmetrics version < 1.0.0:', torchmetrics.__version__)
        #     metrics_kwargs = {}
        # else:
        #     print('torchmetrics version > 1.0.0:', torchmetrics.__version__)
        if self.multilabel:
            metrics_kwargs = dict(num_labels=n_classes, task='multilabel')
        else:
            metrics_kwargs = dict(num_classes=n_classes, task='multiclass')
        metrics[phase] = {'acc':torchmetrics.Accuracy(**metrics_kwargs),
                          'f1':torchmetrics.F1Score(average="micro", **metrics_kwargs)
                          }
        # if self.n_classes==2:
        metrics[phase]['auc'] = torchmetrics.AUROC(average="macro", **metrics_kwargs)

    def _get_targets(self, batch):
        target = batch[self._target_key]
        if self.multilabel:
            target = target.to(torch.float32)
        else:
            if len(target.shape)==2 and target.shape[1]>1: #onehot
                target = target.argmax(1)
        return target


    def predict_step(self, batch, batch_idx, dataloader_idx = None):
        result = self(batch)
        logits = self._get_logits(result)
        if self.multilabel:
            probs = torch.sigmoid(logits)
        else:
            probs = logits.softmax(dim=-1)
        return probs


class ClfGroupLoss(object):
    """ Additional loss for additional group, combining several outputs """
    def __init__(self, group_weights, name='group'):
        """ name: name of the group (manily for the abnormal class).
        group_weights iterable, giving for the class-weights.
        Uses BCEWithLogitsLoss. (not applicable for multiple groups with crossentropy)
        """
        self.name = name
        self.group_weights = np.array(group_weights)
        if 0 not in group_weights:
            raise ValueError('at least one class has to have 0 weight (negative group)')
        self.pos_class_ind = np.where(self.group_weights)[0]
        self.criterion = nn.BCEWithLogitsLoss()

    def set_module(self, modul:ClfModule):
        self.modul = modul
        # self.modul._add_metrics(self.get_metric_names()[0])

    def get_metric_names(self, phase):
        return [f'{phase}_{self.name}']

    def get_metrics(self, phase):
        phase = self.get_metric_names(phase)[0]
        metrics = {}
        n_classes = 1
        metrics[phase] = {'acc':torchmetrics.Accuracy(num_classes=n_classes),
                          # 'f1':torchmetrics.F1(num_classes=n_classes, average="micro"),
                          'f1':torchmetrics.classification.f_beta.F1Score(num_classes=n_classes, average='micro'),
                          'auc':torchmetrics.AUROC(num_classes=n_classes, average="macro"),
        }
        return metrics

    def _create_targets(self, n, device, positive=False):
        return torch.ones((n), dtype=torch.long, device=device)*int(positive)

    def __call__(self, batch, batch_idx, loss, result, x, y, phase):
        logits = self.modul._get_logits(result)
        self._batch_size = len(logits)
        gw = torch.tensor(self.group_weights, dtype=torch.float32).to(logits.device)
        grouped_logits = torch.matmul(logits, gw)#.to(torch.float32)
        targets = batch[self.modul._target_key]
        grouped_targets = torch.matmul(targets.to(torch.float32), gw)
        grouped_targets[grouped_targets>0] = 1
        group_loss = self.criterion(grouped_logits, grouped_targets)
        loss+=group_loss
        self.modul.log(f"{phase}_{self.name}_loss_step", group_loss, batch_size=len(logits))
        self.modul._aggregate_loss(group_loss, batch_idx, phase=f'{phase}_{self.name}')
        self.modul._compute_metrics_step(target=grouped_targets.to(torch.int64), logits=grouped_logits, phase=f'{phase}_{self.name}')
        return loss



class RegrModule(ModuleBase):
    def __init__(self, loss=None, epoch_metrics=['mse'], **kwargs):
        """ loss: name of the loss, dictionary(loss=Name,etc) or None """
        if loss is None or len(loss)==0:
            loss = {'name':'mse'}
        self.loss_name = loss['name']
        # self.cuts = loss.pop('cuts',None).astype(float)

        if self.loss_name not in ['mse']: raise ValueError('unknown loss %s' % self.loss_name)

        super().__init__(loss=loss, epoch_metrics=epoch_metrics, **kwargs)

    def _get_targets(self, batch):
        targ = batch[self._target_key]
        return targ

    def _compute_loss(self, logits, y, phase):
        if not 'train' in phase and len(y)==1 and self.loss_name=='coxph':
            return None #would otherwise always log that loss can not be computed, less important for valid.
        if y.dtype in [torch.int, torch.int64, torch.double]:
            y = y.to(torch.float)
        return self.criterion(logits.squeeze(), y.squeeze())

    def _post_process(self, logits, target):
        # if logits.dtype==torch.float16:
        #     logits = logits.to(torch.float32)
        # else: raise ValueError('unknown loss %s' % self.loss_name)
        return logits, target

    def _add_metrics(self, phase):
        pass

    def _get_targets(self, batch):
        target = batch[self._target_key]
        return target

    def predict_step(self, batch, batch_idx, dataloader_idx = None):
        result = self(batch)
        logits = self._get_logits(result)
        return logits