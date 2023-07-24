from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from pytorch_lightning import Callback
try:
    from pytorch_lightning.loggers import LightningLoggerBase
    from pytorch_lightning.callbacks import ProgressBar
except:
    from pytorch_lightning.loggers.base import LightningLoggerBase
    # from pytorch_lightning.loggers import Logger as LightningLoggerBase
    from pytorch_lightning.callbacks import RichProgressBar as ProgressBar

import matplotlib.pyplot as plt

from pytorch_lightning.utilities.distributed import rank_zero_only
import pytorch_lightning as pl


from wsilearn.utils.cool_utils import is_iterable, lists_disjoint, list_intersection

#FIXME: Rework for lightning

class EpochCsvLogger(LightningLoggerBase):
    """ Logs into a single file taking the last value per epoch.
    (Note that with pl 1.6 sometimes when logging manuall the train epoch loss, the epoch-value seems incremented.
    In this case logging the epoch manually as i.e. epoch_col 'Epoch' solves the issue. """
    def __init__(self, out_dir, epoch_col='epoch', ignore_containing='_step', final_fct=None, **kwargs):
        self.out_path = Path(out_dir)/'history.csv'
        self.epoch_col = epoch_col
        super().__init__(**kwargs)
        self.entries = defaultdict(dict)
        if self.out_path.exists():
            df = pd.read_csv(str(self.out_path))
            entries = df.to_dict('records')
            for entry in entries:
                self.entries[int(entry[epoch_col])] = entry

        self.epoch = None
        self.entry = {}
        self._pending = False

        if ignore_containing is None:
            ignore_containing = []
        if not is_iterable(ignore_containing):
            ignore_containing = [ignore_containing]
        self.ignore_containing=ignore_containing

        self.final_fct = final_fct

    def _filter_metrics(self, metrics):
        if self.ignore_containing is None or len(self.ignore_containing)==0:
            return metrics
        ignore = []
        for k in metrics.keys():
            for ign in self.ignore_containing:
                if ign in k:
                    ignore.append(k)
                    break
        m = {k:v for k,v in metrics.items() if k not in ignore}
        return m

    @property
    def name(self):
        return "EpochCsvLogger"

    @property
    def version(self):
        # Return the experiment version, int or str.
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        # print('hyperparams:',params)
        # self.entry.update(params)
        pass

    def agg_and_log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """ without overwriting this method the last epoch is not logged """
        return self.log_metrics(metrics, step)

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        # print('\nstep %d:' % step, metrics, flush=True)
        self.epoch = metrics.get(self.epoch_col, self.epoch)
        metrics = self._filter_metrics(metrics)
        self._rename_lr_cols(metrics)
        # metrics = self._flatten_dict(metrics)
        if self.epoch is not None and len(metrics)>0:
            metrics['epoch'] = int(self.epoch)
            self.entries[self.epoch].update(metrics)
            self._pending = True

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        if self._pending:
            entries = list(self.entries.values())
            if len(entries[-1])<=1:
                entries = entries[:-1]
            if len(entries)>=2 and 'lr' in entries[-2] and not 'lr' in entries[-1]:
                #the last lr-value is not logged, so here fill with the last available value
                entries[-1]['lr'] = entries[-2]['lr']
            if len(entries)>0:
                df = pd.DataFrame(entries)
                self._rename_lr_cols(df)
                df = df.sort_values(self.epoch_col)
                df.to_csv(str(self.out_path), index=None)
                self._pending = False

    def _rename_lr_cols(self, metrics):
        """ if there is only one learning rate (starts with lr-) save it just as lr """
        lr_cols = [str(col) for col in metrics.keys() if str(col).startswith('lr-')]
        if len(lr_cols)==1:
            lr = metrics.pop(lr_cols[0])
            metrics['lr'] = lr

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        if self.final_fct is not None:
            self.final_fct()

    def experiment(self):
        return None

class HistoryPlotCallback(Callback):
    def __init__(self, csv_path, plot_filename='history.jpg', x_col='epoch', logy=False,
                 plots_keys=[['lr', 'train_loss', 'val_loss', 'val_loss_avg', 'train_reg_loss', 'train_inv_loss', 'train_group_loss', 'train_inst_loss'],
                             ['lr', 'train_acc', 'val_acc', 'train_auc', 'val_auc', 'train_inst_acc', 'train_inst_auc']+
                             ['train_cind', 'val_cind', 'train_group_auc', 'val_group_auc', 'train_auc_avg', 'val_auc_avg']
                             ],
                 plot_key_prio_override={'train_auc':['train_acc'],'val_auc':['val_acc'], 'train_auc_avg':['train_acc_avg'],
                                         'val_auc_avg':['val_acc_avg']}
                 ):
        """ plot_keys: list of two lists for the keys for the first and second plot
        plot_key_prio_override: {important_key:[unimportant keys]} - if important key is present, the unimportant keys
        wont be plotted, e.g. when auc is present dont plott acc """
        super().__init__()
        self.csv_path = Path(csv_path)
        self.plot_filename = plot_filename
        self.plots_keys = plots_keys
        self.plot_key_prio_override = plot_key_prio_override
        self.x_col = x_col
        self.logy = logy

    def __call__(self):
        if not self.csv_path.exists():
            print('History plotting not possible, %s not found' % str(self.csv_path))
            return

        df = pd.read_csv(str(self.csv_path), index_col=self.x_col)
        cols = [str(c) for c in df.columns]

        out_dir = Path(self.csv_path).parent
        self.plot_path = out_dir / self.plot_filename

        found_plots_keys = []
        for pkeys in self.plots_keys: #dont try to plot diagramm if all keys for the diagram are missing
            inters = list_intersection(pkeys, cols)
            if (len(inters)-('lr' in inters))>=1:
            # if not lists_disjoint(pkeys, cols):
                found_plots_keys.append(pkeys)

        n_plots = len(found_plots_keys)
        legs = []
        try:
            if len(df)<=1:
                return
            fig, axs = plt.subplots(n_plots, 1, figsize=(10,min(5*n_plots,20)))
            if n_plots==1: axs = [axs]
            for i, (plot_keys, ax) in enumerate(zip(found_plots_keys, axs)):
                plot_keys = plot_keys.copy()
                # if 'lr' in plot_keys: plot_keys.remove('lr')
                if 'lr' in plot_keys:
                    # ax.set_ylabel('auc')
                    ax_lr = ax.twinx()
                    #df['lr'].plot(ax=ax_lr, logy=True, grid=True, color='k', secondary_y=True)
                    df['lr'].plot(drawstyle="steps", ax=ax_lr, logy=True, grid=True, color='grey', legend=True)
                    leg = ax_lr.legend(loc=0)
                    # leg.set_draggable(True) #### for manual plotting ####
                    legs.append(leg)
                    ax_lr.set_ylabel("lr", rotation=0)
                    # ax_lr.set_xticks(df.index)
                    # ax_lr.set_xticklabels(df.index)
                    plot_keys.remove('lr')
                for key in plot_keys.copy():
                    if key not in df:
                        # print('Cant plot missing %s!' % key)
                        plot_keys.remove(key)
                for prio_key, ignore_keys in self.plot_key_prio_override.items():
                    if prio_key in plot_keys:
                        for ignore_key in ignore_keys:
                            if ignore_key in plot_keys:
                                plot_keys.remove(ignore_key)
                try:
                    if len(plot_keys)>0:
                        df[plot_keys].plot(ax=ax, grid=True, legend=True, logy=self.logy)
                except Exception as e:
                    print('Plotting %s failed: %s' % (str(plot_keys), str(e)))
                if i==0:
                    # leg = ax.legend(loc='upper center')
                    leg = ax.legend(loc='best')
                else:
                    leg = ax.legend(loc='best')
                # leg.set_draggable(True)  ####### for manual plotting #####
                legs.append(leg)
                ax.grid('on', which='minor', axis='y')
                # ax.set_xticks(df.index)
                # ax.set_xticklabels(df.index)
            # ax.legend('upper center')
            # df[plot_keys].plot(secondary_y=['lr'], mark_right=False, ax=ax, grid=True)
            ########## for manual plotting
            # DraggableLegendTwinxWorkaround(legs)
            # plt.show()
            #########
            plt.savefig(str(self.plot_path), bbox_inches='tight')
            plt.close()
        except Exception as e:
            print('Plotting failed: %s' % str(e))


class MeterlessProgressBar(ProgressBar):
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.dynamic_ncols = False
        bar.ncols = 0
        return bar