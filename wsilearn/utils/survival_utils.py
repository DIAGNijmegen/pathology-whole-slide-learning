#https://andre-rendeiro.com/2016/01/04/survival_analysis_with_lifelines_part1
import numpy
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from lifelines import NelsonAalenFitter, KaplanMeierFitter, CoxPHFitter
import itertools
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from matplotlib.offsetbox import AnchoredText

from wsilearn.utils.cool_utils import is_iterable
from wsilearn.dataconf import DataConf
from wsilearn.utils.df_utils import print_df, df_dropna
from wsilearn.utils.eval_utils import DataEvaluator


def _count_event_suffix(df, event_col):
    n = len(df)
    n1 = (df[event_col]==1).sum()
    return ' (%s/%s events)' % (n1, n)

def coxph(df, event_col='Event_OS', time_col='OS', markers=None, verbose=False):
        if markers is not None:
            if not is_iterable(markers):
                markers = markers.split(',')
                markers = [m.strip() for m in markers]
            df = df[[event_col, time_col] + markers].copy()
        print('computing concordance index with CoxPHFitter on df with columns %s' % str(df.columns))
        n = len(df)
        df = df.dropna()
        if n != len(df):
            print('%d/%d (%d dropped)' % (len(df), n, n - len(df)))
        cph = CoxPHFitter().fit(df, time_col, event_col)
        if verbose:
            cph.print_summary()
        cind = cph.concordance_index_
        pval = cph.log_likelihood_ratio_test().p_value
        coef = cph.params_.values[0]
        other_cols = [str(c) for c in df.columns if str(c) not in [time_col, event_col]]
        print('Concordance index for %s: %f, p=%f, coef=%.4f, coef_exp=%.4f' % (
            str(other_cols), cind, pval, coef, np.exp(coef)))
        # concordance_index(df['T'], -cph.predict_partial_hazard(df), df['E'])
        expect = cph.predict_expectation(df)
        return cind, pval, expect

def compute_concordance(df, event_col, time_col, marker_col=None, verbose=False):
    cind_direct = None
    if marker_col is not None:
        if not is_iterable(marker_col):
            marker_col = [marker_col]
        df = df[[event_col, time_col] + marker_col].copy()
        n_df = len(df)
        df = df_dropna(df, marker_col)
        if len(df)!=n_df:
            print('concordance for %d/%d due to missing %s values' % (len(df), n_df, str(marker_col)))

        if len(marker_col)==1:
            cind_direct = concordance_index(event_times=df[time_col], predicted_scores=df[marker_col], event_observed=df[event_col])
            print('cind direct=%.4f' % (cind_direct))

    print('computing concordance index with CoxPHFitter on df with columns %s' % str(df.columns))
    n = len(df)
    df = df.dropna()
    if n != len(df):
        print('%d/%d (%d dropped)' % (len(df), n, n - len(df)))
    cph = CoxPHFitter().fit(df, time_col, event_col)
    if verbose:
        cph.print_summary()
    cind = cph.concordance_index_
    pval = cph.log_likelihood_ratio_test().p_value
    if cind_direct is not None and cind_direct!=cind:
        raise ValueError('cind_direct=%f, but cind via coxph=%f' % (cind_direct, cind))
    coef = cph.params_.values[0]
    other_cols = [str(c) for c in df.columns if str(c) not in [time_col, event_col]]


    print('Concordance index for %s: %f, p=%f, coef=%.4f, coef_exp=%.4f' % (
        str(other_cols), cind, pval, coef, np.exp(coef)))
    # concordance_index(df['T'], -cph.predict_partial_hazard(df), df['E'])
    return cind, pval

def plot_kaplan_meier(df, marker_col, time_col, event_col,
                      plot_all=False, label_all='All', title=None, ax=None, show=False, **kwargs):
    return plot_survival(df, marker_col, time_col, event_col, fitter='survival', discrete=False,
                         plot_all=plot_all, label_all=label_all, title=title, ax=ax, show=show, **kwargs)

""" adapted from https://andre-rendeiro.com/2016/01/04/survival_analysis_with_lifelines_part1 """
def plot_survival(df, marker_col, time_col, event_col, fitter='survival', cutoff='median', discrete=False, save_path=None,
                  plot_all=False, label_all='All', title=None, title_cind=True, ax=None, show=False,
                  # value_label_map=None,
                  p_loc='center', leg_loc=None, ymin=0):
    if fitter=='survival':
        fitter = KaplanMeierFitter()
        ylim = (ymin, 1.05)
        suffix = 'Survival'
    elif fitter=='hazard':
        fitter = NelsonAalenFitter()
        ylim = None
        suffix = 'Hazard'
    else:
        raise ValueError('unknown fitter %s' % fitter)

    cutoff_map = {'median':np.median, 'mean':np.mean}
    cutoff_fct = cutoff_map.get(cutoff, cutoff)

    # print('plotting %s for %s' % (fitter, marker_col))
    df = df[[time_col, event_col, marker_col]]
    dfna = df[df.isna().any(axis=1)]
    if len(dfna)>0:
        print('excluding %d/%d without values' % (len(dfna), len(df)))
        df = df.dropna()
    dfo = df
    df = df.copy()

    # if value_label_map is None:
    #     value_label_map = {}
    value_label_map = {}
    cind = concordance_index(event_times=df[time_col], predicted_scores=df[marker_col], event_observed=df[event_col])

    if discrete:
        df[marker_col] = df[marker_col].astype(int)
    else:
        median = cutoff_fct(df[marker_col])
        print('median cutoff for %s: %.4f' % (marker_col, median))
        if df[marker_col].min()<median:
            value_label_map[0] = '<%.4f' % median
            value_label_map[1] = '>=%.4f' % median
            df[marker_col] = df[marker_col]>=median
        else:
            value_label_map[0] = '<=%.4f' % median
            value_label_map[1] = '>%.4f' % median
            df[marker_col] = df[marker_col]>median
        df[marker_col] = df[marker_col].astype(int)

        # if (df[marker_col] % 1  != 0).any(): #floats?
        #     df = df.copy()
        #     if np.array_equal(df[marker_col], df[marker_col].astype(int)):
        #         df[marker_col] = df[marker_col].astype(int)
        #     else:
        #         median = cutoff_fct(df[marker_col])
        #         print('median cutoff for %s: %.4f' % (marker_col, median))
        #         if 0 not in value_label_map:
        #             value_label_map[0] = '<=%.4f' % median
        #         if 1 not in value_label_map:
        #             value_label_map[1] = '>%.4f' % median
        #         df[marker_col] = df[marker_col]>median
        #         df[marker_col] = df[marker_col].astype(int)
        # elif 'float' in str(df.dtypes[marker_col]):
        #     df = df.copy()
        #     df[marker_col] = df[marker_col].astype(int)


    # T = [i.days / float(30) for i in df[time_col]]  # duration of patient following
    # events:
    # True for observed event (death);
    # else False (this includes death not observed; death by other causes)
    # C = [True if i is not pd.NaT else False for i in df["patient_death_date"]]

    fig = None
    if ax is None:
        fig, ax = plt.subplots(1)

    if plot_all:
        # all together
        fitter.fit(df[time_col], event_observed=df[event_col], label=label_all+_count_event_suffix(df, event_col))
        fitter.plot_survival_function(ax=ax, show_censors=True)

    # Filter feature types which are nan
    labels = list(df[marker_col].unique())
    # label = label[~np.array(map(pd.isnull, label))]

    # Separately for each class
    for value in labels:
        value_label = value_label_map.get(value, str(value))
        # print('fitting for value', value)
        dfv = df[df[marker_col] == value]
        fitter.fit(dfv[time_col], event_observed=dfv[event_col], label=value_label+_count_event_suffix(dfv, event_col))
        fitter.plot_survival_function(ax=ax, show_censors=True)

    if ylim is not None:
        ax.set_ylim(*ylim)

    # Test pairwise differences between all classes
    p_vals = []; p_vals_str = []
    combis = list(itertools.combinations(labels, 2))
    for a, b in combis:
        dfa = df[df[marker_col] == a]
        dfb = df[df[marker_col] == b]
        p = logrank_test(
            dfa[time_col], dfb[time_col],
            event_observed_A=dfa[event_col],
            event_observed_B=dfb[event_col]).p_value
        # see result of test with p.print_summary()
        if len(combis)>1:
            p_vals_str.append("p-value '" + " vs ".join([str(a), str(b)]) + "': %f" % p)
        else:
            p_vals_str.append('p = %f' % p)
        p_vals.append(p)
    p_str = "\n".join(p_vals_str)
    print(p_str)
    # Add p-values as anchored text
    prop=dict(fontweight="bold") if p <= 0.05 else None
    ax.add_artist(AnchoredText(p_str, loc=p_loc, frameon=False, prop=prop))

    if title is None:
        title = '%s %s' % (marker_col, suffix)

    if title_cind:
        title += ' cind=%.3f' % (cind)
    ax.set_title(title)

    ax.grid(True)

    if leg_loc is not None:
        ax.legend(loc=leg_loc)
    if show:
        plt.tight_layout()
        plt.show()

    if fig is not None and save_path is not None:
        fig.tight_layout()
        fig.savefig(str(save_path), bbox_inches="tight")

    if len(p_vals)==1:
        p_vals = p_vals[0]

    # ax.get_legend().set_draggable(state=1)
    return cind, p_vals


class SurvEvaluator(DataEvaluator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _evaluate(self, data_conf:DataConf, out_dir:Path=None):
        # preds = data_conf.get_out()
        # targets = data_conf.get_targets()

        # event = targets[:,0]
        # os = targets[:,1]
        # cind = concordance_index(preds, os, event)


        out_cols = data_conf.get_out_cols()
        if len(out_cols)==1: #coxph
            out_col = data_conf.get_out_cols()[0]
        else: #nll, for simplicity just summing all cuts
            data_conf.df[data_conf.out_col] = data_conf.df[out_cols].sum(axis=1)/len(out_cols)
            out_col = data_conf.out_col

        if len(np.unique(data_conf.get_targets_surv_events()))<2:
            print('survival plot needs both events')
            from lifelines.utils import concordance_index
            cind = concordance_index(event_times=data_conf.get_targets_surv_durations(),
                                     predicted_scores=data_conf.df[out_col],
                                     event_observed=data_conf.get_targets_surv_events())

            return dict(cind=cind, p=-1)
        else:
            km_path = Path(out_dir)/'km.jpg'
            cind, p = plot_survival(data_conf.df, out_col, data_conf.get_surv_duration_col(),
                          event_col=data_conf.get_surv_event_col(), save_path=km_path, show=False)

            metrics = dict(cind=cind, p=p)
            return metrics