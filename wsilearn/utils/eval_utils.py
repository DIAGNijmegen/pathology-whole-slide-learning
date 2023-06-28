import traceback, sklearn
from collections import OrderedDict
from typing import Iterable
from copy import deepcopy

import matplotlib
from matplotlib.ticker import LinearLocator
from scipy.stats import chi2_contingency
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, \
    balanced_accuracy_score, recall_score, cohen_kappa_score, roc_auc_score, roc_curve, \
    multilabel_confusion_matrix, brier_score_loss
from sklearn.preprocessing import label_binarize

from wsilearn.utils.cool_utils import *
import pandas as pd

from wsilearn.utils.cool_utils import is_df, is_iterable, is_string
from wsilearn.dataconf import DataConf
from wsilearn.utils.pycm_utils import pycm_metrics
from wsilearn.utils.df_utils import print_df, print_df_trac
from wsilearn.utils.parse_utils import parse_string

import warnings
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

########## NEEDS CLEANUP AND REFACTORING1##################

class DataConfEvaluatorCallback(object):
    def evaluate(self, data_conf:DataConf, out_dir):
        raise ValueError('implement!')

class DataEvaluator(object):
    def __init__(self, callbacks:Iterable[DataConfEvaluatorCallback]=[], max_category_groups=16):
        if callbacks is None:
            callbacks = []
        self.callbacks = callbacks
        self.max_category_groups = max_category_groups

    def evaluate_categories(self, data_conf:DataConf, out_dir, n=''):
        metrics_map = {}
        cat_groups = data_conf.get_category_groups(n=n)
        uni_cat_groups = [cg for cg in cat_groups if len(cg)==1]
        if len(cat_groups)>self.max_category_groups:
            #only evaluate single category groups
            print('only evaluating %d single category groups since otherwise too many categories (%d)' %\
                  (len(uni_cat_groups), len(cat_groups)))
            cat_groups = uni_cat_groups

        for categories in cat_groups:
            cconf = deepcopy(data_conf)
            cconf.select(category=categories, category_n=n)
            cname = '_'.join(categories)
            cat_out_dir = Path(out_dir)/cname
            try:
                cat_results = self.evaluate(cconf, cat_out_dir)
            except Exception as ex:
                print('evaluating categories %s failed' % str(categories))
                print(ex)
                continue
            metrics_map[cname] = cat_results
            # cat_results = self.eval(slide_pred_df, save_plots=save_plots, detail=detail, out_dir=cat_out_dir)
            # metrics_map[prefix+'_'+cname] = cat_results


        return metrics_map

    def evaluate_all(self, data_conf, out_dir):
        """ evaluates the splits and categories (if present) """
        metrics = self.evaluate(data_conf, out_dir=out_dir)
        for category_n in ['', 2, 3]:
            if data_conf.has_categories(n=category_n):
                out_dir_c = Path(out_dir)/f'categories{category_n}'
                pcmetrics = self.evaluate_categories(data_conf, out_dir_c, n=category_n)
                metrics.update(pcmetrics)
                self._evaluate_categories_post(data_conf, out_dir, n=category_n)
        if out_dir is not None:
            out_path = Path(out_dir) / 'results.json'
            write_json_dict(path=out_path, data=metrics)
        return metrics

    def evaluate(self, data_conf, out_dir):
        if out_dir is not None:
            out_dir = Path(out_dir)
            ensure_dir_exists(out_dir)
        result = self._evaluate(data_conf, out_dir) #implement also as callback?
        for cb in self.callbacks:
            cb.evaluate(data_conf, out_dir)
        return result

    def _evaluate(self, data_conf, out_dir):
        raise ValueError('implement evaluate')

    def _evaluate_categories_post(self, data_conf, out_dir, n=''):
        """ hook to create a per-category 'confusion matrix' or similar overview after the main evaluation """
        pass

    def get_pred_path(self, out_dir):
        return Path(out_dir)/'predictions.csv'

class ClfEvaluator(DataEvaluator):
    def __init__(self, class_names, use_validation_thresholds=True, clf_thresholds=None,
                 tpr_weight=None, **kwargs):
        super().__init__(**kwargs)
        self.class_names = class_names
        self.n_classes = len(class_names)

        self.use_clf_thresholds = use_validation_thresholds
        self.clf_thresholds = clf_thresholds
        self.tpr_weight = tpr_weight
        print('created ClfEvaluator for classes %s, thresholds %s and tpr_weight %s' %\
              (str(class_names), str(clf_thresholds), str(tpr_weight)))

    def _evaluate(self, data_conf:DataConf, out_dir:Path=None):
        probs = data_conf.get_out()

        pred_out_path = None
        if out_dir is not None:
            pred_out_path = self.get_pred_path(out_dir)

        if data_conf.is_clf():
            targets = data_conf.get_targets_labels()
        else:
            targets = data_conf.get_targets() #hot encoded
        # if data_conf.train_type == TrainType.clf and probs.shape[1]==2:
        #     probs = probs[:,1]
        #     if targets.shape[1]==2: #tmp fix
        #         targets = targets[:,1]

        if self.use_clf_thresholds:
            if self.clf_thresholds is None:
                if data_conf.get_splits()!=['validation']:
                    raise ValueError('needs to compute thresholds for multilabel validation first')
                self.find_clf_thresholds(data_conf)

        cmetrics = clf_metrics2(targets, probs=probs, save_plots=True, thresh_tpr=self.clf_thresholds,
                               class_names=self.class_names, detail=True, silent=True, out_dir=out_dir)
        pred = cmetrics.pop('pred')

        # pred = None
        # if self.use_clf_thresholds and (data_conf.train_type == TrainType.multilabel):
        #     #needs to find optimal clf_threshs first
        #     if self.clf_thresholds is None:
        #         if data_conf.get_splits()!=['validation']:
        #             raise ValueError('needs to compute thresholds for multilabel validation first')
        #         self.find_clf_thresholds(data_conf)
        #     pred = threshold_probs(probs, self.clf_thresholds)
        # cmetrics, pred = clf_metrics(targets, pred=pred, probs=probs, save_plots=True,
        #                        class_names=self.class_names, multilabel=data_conf.train_type==TrainType.multilabel,
        #                        detail=False, ret_pred=True, silent=True, out_dir=out_dir)

        data_conf.add_pred(pred)
        if out_dir is not None:
            data_conf.to_csv(pred_out_path)
            create_simple_short_info_file(out_dir, cmetrics)
        if self.clf_thresholds is not None:
            for ci, thresh in enumerate(self.clf_thresholds):
                if self.class_names is not None:
                    ci = self.class_names[ci]
                cmetrics[f'clf_thresh_{ci}'] = thresh
            cmetrics['clf_thresholds'] = self.clf_thresholds


        return cmetrics


    def find_clf_thresholds(self, data_conf:DataConf, split='validation'):
        """ finds the classification cutoffs per class using roc_auc analysis """
        if split is not None:
            data_conf = deepcopy(data_conf)
            data_conf.select(split=split)

        prob_cols = data_conf.get_out_cols()
        targets = data_conf.get_targets()
        probs = data_conf.get_out()
        threshs = []
        tpr_weight = self.tpr_weight
        if not is_iterable(tpr_weight):
            tpr_weight = [tpr_weight]*probs.shape[1]
        for i,pcol in enumerate(prob_cols):
            tari = targets[:,i]
            probsi = probs[:,i]
            resulti = roc_analysis(tari, probsi, tpr_weight=tpr_weight[i], detail=False)
            threshi = resulti['thresh']
            threshs.append(threshi)
        self.clf_thresholds = threshs
        print('found %s thresholds: %s' % (split, str(threshs)))
        return np.array(threshs)

    def _evaluate_categories_post(self, data_conf:DataConf, out_dir, n=''):
        """ creates a confusion matrix per category assuming that there is only one class per category """
        if out_dir is None:
            return
        if data_conf.is_multilabel():
            return

        pred_out_path = self.get_pred_path(out_dir)
        dfp = pd.read_csv(pred_out_path)
        categories = data_conf.get_categories(n=n)
        if len(categories)==1:
            return

        classes = data_conf.get_target_cols()
        cat_matrix = np.zeros((len(categories), len(classes)))
        for c,cat in enumerate(categories):
            df = data_conf.select(dfp, category=cat, category_n=n)
            labels = data_conf.get_labels(df)
            if len(set(labels))==0:
                print('category evaluation ERROR! no labels for category %s' % cat)
                return
            elif len(set(labels))>1:
                return #only works when there is only one label per category
            preds = data_conf.get_pred_labels(df)
            if len(preds)!=len(labels): raise ValueError('different number of labels (%d) and predictions (%d)'
                                                         % (len(labels), len(preds)))
            for l in range(data_conf.n_classes):
                cat_matrix[c,l] = np.sum(preds==l)
        save_path = Path(out_dir)/'cm_categories.jpg'
        plot_confusion_matrix(cat_matrix, classes, ylabels=categories, show=False, normalize=True,
                              save_path=save_path)
        plot_confusion_matrix(cat_matrix, classes, ylabels=categories, show=False,
                              save_path=str(save_path).replace('.jpg', '_abs.jpg'), normalize=False)


class RegrEvaluator(DataEvaluator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _evaluate(self, data_conf:DataConf, out_dir:Path=None):


        out_cols = data_conf.get_out_cols()
        if len(out_cols)==1: #coxph
            out_col = data_conf.get_out_cols()[0]
        else: #nll, for simplicity just summing all cuts
            data_conf.df[data_conf.out_col] = data_conf.df[out_cols].sum(axis=1)/len(out_cols)
            out_col = data_conf.out_col

        targets = data_conf.get_targets()
        preds = data_conf.get_out()
        mse = sklearn.metrics.mean_squared_error(targets, preds)
        rmse = sklearn.metrics.mean_squared_error(targets, preds, squared=False)
        r2 = sklearn.metrics.r2_score(targets, preds)

        metrics = dict(mse=mse, rmse=rmse, r2=r2)
        return metrics

def specificity_true_negative_rate_bin(labels, preds):
    cm = confusion_matrix(labels, preds)
    tn = cm[0][0]
    fn = cm[1][0]
    tp = cm[1][1]
    fp = cm[0][1]

    specificity = tn / (tn + fp)

    return specificity

def false_positive_rate_bin(labels, preds):
    cm = confusion_matrix(labels, preds)
    TN = cm[0][0]
    FN = cm[1][0]
    TP = cm[1][1]
    FP = cm[0][1]

    FPR = FP / (FP + TN)

    return FPR

def false_negative_rate_bin(labels, preds):
    cm = confusion_matrix(labels, preds)
    TN = cm[0][0]
    FN = cm[1][0]
    TP = cm[1][1]
    FP = cm[0][1]

    FNR = FN/(FN+TP)

    return FNR

def binary_clf_pred_metrics(labels, pred, detail=True, kappa_weights=None):
    cm = confusion_matrix(labels, pred)
    f1 = f1_score(labels, pred)
    acc = accuracy_score(labels, pred)
    bacc = balanced_accuracy_score(labels, pred)
    n = len(pred)
    n_pos = np.sum(labels==1)
    tpr = recall_score(labels, pred)
    fpr = false_positive_rate_bin(labels, pred)
    fnr = false_negative_rate_bin(labels, pred)
    kappa = cohen_kappa_score(labels, pred, weights=kappa_weights)
    tnr = specificity_true_negative_rate_bin(labels, pred)
    results = dict(cm=cm, f1=f1, acc=acc, bacc=bacc, n_pos=n_pos, n=n,  tpr=tpr, fpr=fpr, tnr=tnr, fnr=fnr, kappa=kappa)
    return results

def _probs1d(y_true, probs):
    if len(np.unique(y_true))<=2 and len(probs.shape)==2 and probs.shape[1]==2:
        #if binary and probs 2dimensional, sklearn wants the score of the class with the greater label
        probs = probs[:,1]
    return probs

def roc_auc(y_true, probs):
    probs = _probs1d(y_true, probs)
    if np.isnan(probs).any():
        raise ValueError('NaN in probs!')
    auc = roc_auc_score(y_true, probs)
    return auc

def _align_grid(ax1, ax2, N=None):
    if N is None:
        N = len(ax1.yaxis.get_major_ticks())
    ax1.yaxis.set_major_locator(LinearLocator(N))
    ax2.yaxis.set_major_locator(LinearLocator(N))

def threshold_probs(probs, thresh):
    result = probs>=thresh # >= !
    # if is_ndarray(result):
    result = result.astype(np.uint8)
    return result

def bin_chi_square_test_df(df, target_col, pred_col, thresh=None, low_label='low', high_label='high',
                           print_trac=False, print_latex=False, verbose=False):
    df = df.copy()
    if thresh is not None:
        df[pred_col] = df[pred_col] > thresh
    if low_label is not None:
        df.replace({pred_col: {0: low_label, 1: high_label}}, inplace=True)
    df.replace({target_col: {0: 'no '+target_col, 1: target_col}}, inplace=True)

    df_grouped = df.groupby([target_col, pred_col]).size().reset_index().rename(columns={0: 'count'})
    # print_df(df_pcr_grouped)
    df_matr = df_grouped.pivot_table(columns=pred_col, index=target_col, values='count')
    df_matr.fillna(0, inplace=True)
    stat, p, dof, expected = chi2_contingency(df_matr)
    if verbose:
        print('chi square test for %s stat: %.3f, p: %.5f' % (target_col, stat, p))
        print_df(df_matr)
    if print_trac:
        print_df_trac(df_matr, index_name=True)
    if print_latex:
        print(df_matr.to_latex())

    return df_matr, p

def bin_chi_square_test(labels, probs, thresh, pred_col='Response', low_label='low', high_label='high', **kwargs):
    if not is_ndarray(labels):
        labels = np.array(labels)
    if not is_ndarray(probs):
        probs = np.array(probs)
    if len(probs.shape)==2:
        assert probs.shape[1]==1
        probs = probs.ravel()
    if len(labels.shape)==2:
        assert labels.shape[1]==1
        labels = labels.ravel()

    target_col = 'labels'
    df = pd.DataFrame({target_col:labels, pred_col:probs})

    # df[pred_col] = probs > thresh
    # df.replace({pred_col:{0:low_label, 1:high_label}}, inplace=True)
    df = df[[target_col,pred_col]]

    return bin_chi_square_test_df(df, target_col, pred_col, thresh=thresh, low_label=low_label, high_label=high_label, **kwargs )
    # df_grouped = df.groupby([target_col, pred_col]).size().reset_index().rename(columns={0: 'count'})
    # # print_df(df_pcr_grouped)
    # df_matr = df_grouped.pivot_table(columns=target_col, index=pred_col, values='count')
    # df_matr.fillna(0, inplace=True)
    # print_df(df_matr)
    # stat, p, dof, expected = chi2_contingency(df_matr)
    # print('chi square test for %s stat: %.3f, p: %.5f' % (target_label, stat,  p))

def binary_clf_metrics(labels, probs, thresh=0.5, kappa_weights=None, detail=True):
    if not (0 in np.unique(labels) and 1 in np.unique(labels) and len(np.unique(labels))==2):
        raise ValueError('invalid labels', labels)
    labels = labels.astype(np.uint8)
    pred = threshold_probs(probs,thresh)
    _, p_thresh = bin_chi_square_test(labels, probs, thresh=thresh, verbose=False)
    auc = roc_auc(labels, probs)
    results = dict(auc=auc, thresh=thresh, p_thresh=p_thresh)
    results.update(binary_clf_pred_metrics(labels, pred, detail=detail, kappa_weights=kappa_weights))
    results['pred'] = pred
    return results

def roc_gmean(fpr, tpr, w=None):
    """ w: weighting ratio between sensitivity (tpr) and specificity"""
    #tpr: sensitivity
    #specificity: tnr=1-fpr
    spec = 1-fpr
    if w is None:
        gmeans = np.sqrt(tpr * spec)
    else:
        wsens = w/(w+1)
        wspec = 1-wsens
        gmeans = np.exp(wsens*np.log(tpr+1e-12)+wspec*np.log(spec+1e-12))
    ix = np.argmax(gmeans)
    return gmeans[ix], ix

def roc_youden(fpr, tpr, w=None):
    """
    youden: sensitivity + specificity - 1; tpr: sensitivity, fpr: 1-specificity; specificity = 1-fpr
    youden = sensitivity + (1-fpr) -1 = tpr - fpr
    """
    #unweighted just tpr-fpr
    if w is None:
        y = tpr-fpr
    else:
        spec = 1-fpr
        wsens = w/(w+1)
        wspec = 1-wsens
        y = 2*(wsens*tpr+wspec*spec)-1
    ix = np.argmax(y)
    return y[ix], ix

def roc_analysis(labels, probs, pos_label=None, label='ROC curve', thresh_tpr=None, silent=True, detail=True,
                 tpr_weight=None, youden=False, kappa_weights=None):
    """
    :returns dict with: 'auc', 'thresh', 'p_thresh', 'cm', 'f1', 'acc', 'bacc',
            'n_pos', 'n', 'tpr', 'fpr', 'tnr', 'kappa', 'pred'
                and if details: 'fprs', 'tprs', 'threshs'
    """
    probs = _probs1d(labels, probs)
    fprs, tprs, threshs = roc_curve(labels, probs, pos_label=pos_label)
    #for ill configured problebms or toy-debugging it can be that threshs first value is bigger then 1
    if threshs[0]>1 and max(probs)<1:
        threshs[0] = 1

    # rauc = metrics.auc(fpr, tpr)
    # maximizing the geometric mean
    # https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/
    cut_fct = roc_youden if youden else roc_gmean
    cut_fct_name = str(cut_fct).replace('roc_','')
    if thresh_tpr is None:
        gmean, ix = cut_fct(fprs, tprs, w=tpr_weight)
        thresh = threshs[ix]
        fpr_thresh_val = fprs[ix]
        tpr_thresh_val = tprs[ix]
        if not silent:
            print('%s Best Threshold=%f, %s=%.3f, FPR: %.3f, TPR: %.3f' % \
                  (label, thresh, cut_fct_name, gmean, fpr_thresh_val, tpr_thresh_val))
    else:
        thresh = thresh_tpr
    # else:
    #     tprs_tmp = np.array(tprs).copy()
    #     tprs_tmp[tprs_tmp<thresh_tpr] = 0
    #     gmean, ix = cut_fct(fprs, tprs_tmp)
    #     thresh = threshs[ix]
    #     fpr_thresh_val = fprs[ix]
    #     tpr_thresh_val = tprs[ix]
    #     if fpr_thresh_val >= 0.99 or tpr_thresh_val<thresh_tpr:
    #         thresh_interp = interpolate.interp1d(tprs,threshs)
    #         thresh = thresh_interp(thresh_tpr)
    #         tpr_intrp = interpolate.interp1d(threshs, tprs)
    #         fpr_intrp = interpolate.interp1d(threshs, fprs)
    #         tpr_thresh_val = tpr_intrp(thresh)
    #         fpr_thresh_val = fpr_intrp(thresh)

        # for ix,tpr in enumerate(tprs):
        #     if tpr >= thresh_tpr:
        #         thresh = threshs[ix]
        #         break


    results = {}
    if detail:
        results.update(dict(fprs=fprs, tprs=tprs, threshs=threshs))

    bclf_metrics = binary_clf_metrics(labels, probs, thresh, detail=detail, kappa_weights=kappa_weights)
    # bclf_metrics['thresh_tpr'] = tpr_thresh_val
    # bclf_metrics['thresh_tpr'] = tpr_thresh_val
    results.update(bclf_metrics)
    # pred = probs>thresh[ix]
    # cm = confusion_matrix(labels, pred)
    # f1 = f1_score(labels, pred)
    # acc = accuracy_score(labels, pred)
    # bacc = balanced_accuracy_score(labels, pred)
    # results.update(dict(cm=cm, f1=f1, acc=acc, bacc=bacc, pred=pred))
    return results

def plot_roc_curve_multi(labels, preds, show=False, fig_save_path=None, class_names=None,
                         title='ROC', subplot_size=5, **kwargs):
    n_classes = preds.shape[1]
    if len(labels.shape)==1:
        labels = label_binarize(labels, classes=[i for i in range(n_classes)])

    if n_classes <=4:
        fig, axs = plt.subplots(1, n_classes, figsize=(subplot_size*n_classes, subplot_size+0.5))
    else:
        fig, axs = plt.subplots(n_classes, 1, figsize=(subplot_size+1, n_classes*subplot_size))
    axs = axs.ravel()
    for i in range(n_classes):
        ax = axs[i]
        if class_names is None or len(class_names)==0:
            titlei = 'Class %d' % i
        else:
            titlei = class_names[i]
        plot_roc_curve(labels[:,i], preds[:,i], ax=ax, pos_label=1, title=titlei, **kwargs)

    if title is not None and len(title)>0:
        fig.suptitle(title)
    if show:
        plt.tight_layout()
        plt.show()

    if fig_save_path is not None:
        plt.savefig(str(fig_save_path), bbox_inches='tight', dpi=300)

def plot_roc_curve_multi_one(labels, preds, show=False, fig_save_path=None, class_names=None,
                         title='ROC', thresh_tpr=None, tpr_weight=None, **kwargs):
    n_classes = preds.shape[1]
    if len(labels.shape)==1:
        labels = label_binarize(labels, classes=[i for i in range(n_classes)])


    if not is_iterable(thresh_tpr):
        thresh_tpr = [thresh_tpr]*n_classes
    if not is_iterable(tpr_weight):
        tpr_weight = [tpr_weight]*n_classes

    fig, ax = plt.subplots()
    for i in range(n_classes):
        if class_names is None or len(class_names)==0:
            titlei = 'Class %d' % i
        else:
            titlei = class_names[i]
        if np.sum(labels[:,i])>0: #when multilabel and no labels occurrences of this class - skip
            plot_roc_curve(labels[:,i], preds[:,i], thresh_tpr=thresh_tpr[i], tpr_weight=tpr_weight[i],
                       ax=ax, pos_label=1, label=titlei, **kwargs)

    if title is not None and len(title)>0:
        plt.title(title)
    if show:
        plt.tight_layout()
        plt.show()

    if fig_save_path is not None:
        plt.savefig(str(fig_save_path), bbox_inches='tight', dpi=300)

    return fig, ax

def plot_roc_curve(labels, preds, thresh_tpr=None, fig_save_path=None, csv_save_path=None, pos_label=None, ax=None, ax2=None, lw=2,
                   title='Receiver operating characteristic', label='ROC curve', color=None, plot_thresh=False,
                   plot_best_thresh=True, thresh_lim=None, show=False, centerdot=False, title_fontdict={},
                   leg_outside=False, marker_map={}, linestyle_map={}, xlabel='False Positive Rate',
                   ylabel='True Positive Rate', diag_color='k', **kwargs):
    roc_metrics_results = roc_analysis(labels, preds, pos_label=pos_label, thresh_tpr=thresh_tpr, **kwargs)
    return plot_roc_curve_results(roc_metrics_results, fig_save_path=fig_save_path, csv_save_path=csv_save_path,
                                  ax=ax, ax2=ax2, lw=lw, title=title, label=label, color=color, plot_thresh=plot_thresh,
                                  plot_best_thresh=plot_best_thresh, thresh_lim=thresh_lim, show=show,
                                  centerdot=centerdot, title_fontdict=title_fontdict,
                                  leg_outside=leg_outside, marker_map=marker_map, linestyle_map=linestyle_map,
                                  xlabel=xlabel, ylabel=ylabel, diag_color=diag_color)

def plot_roc_curve_results(roc_metrics_results, fig_save_path=None, csv_save_path=None, ax=None, ax2=None, lw=2,
                   title='Receiver operating characteristic', label='ROC curve', color=None, plot_thresh=False,
                   plot_best_thresh=True, thresh_lim=None, show=False, centerdot=False, title_fontdict={},
                   leg_outside=False, marker_map={}, linestyle_map={}, xlabel='False Positive Rate',
                   ylabel='True Positive Rate', diag_color='dimgrey', diag_lw=1):
    roc_metrics = DictObject(roc_metrics_results)
    rauc, thresh, fpr, tpr = roc_metrics.auc, roc_metrics.thresh, roc_metrics.fpr, roc_metrics.tpr
    fprs, tprs, threshs = roc_metrics.fprs, roc_metrics.tprs, roc_metrics.threshs

    if ax is None:
        fig, ax = plt.subplots()
    elif is_list(ax):
        ax = ax[0]
        fig = None
    else:
        fig = None

    if plot_thresh:
        if ax2 is None:
            ax2 = ax.twinx()
        ax2.set_ylabel('Threshold')

    if plot_best_thresh:
        label_long = label + ' (AUC=%0.3f,thresh=%.3f)' % (rauc, thresh)
    else:
        label_long = label + ' (AUC=%0.3f)' % rauc
    if centerdot:
        label_long = label_long.replace('.','·')

    p = ax.plot(fprs, tprs, color=color, lw=lw, label=label_long, marker=marker_map.get(label,None),
                linestyle=linestyle_map.get(label,None))
    color = p[0].get_color()
    if plot_thresh:
        ax2.plot(fprs, threshs, lw=max(1, lw // 2), ls='--', color=color, alpha=0.7)
        _align_grid(ax, ax2, N=6)
    ax.plot([0, 1], [0, 1], color=diag_color, lw=diag_lw, linestyle='--')

    if plot_best_thresh:
        # ax.plot([fprs[gix]], tprs[gix], marker='o', color=color, markersize=5)
        ax.plot(fpr, tpr, marker='o', color=color, markersize=5)

    if plot_thresh and thresh_lim is not None:
        # thresh[thresh<thresh_lim[0]] = thresh_lim[0]
        # thresh[thresh>thresh_lim[1]] = thresh_lim[1]
        ax2.set_ylim(thresh_lim)
        # ax2.set_aspect('equal')

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    if not plot_thresh: #doesnt work for some reason with second axis here
        ax.set_aspect('equal')
        pass

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontdict=title_fontdict)
    # ax.legend(loc="lower right")

    if leg_outside:
        ax.legend(loc='upper left', ncol=1, bbox_to_anchor=(0, -0.1))  # bbox_to_anchor=(0.5, 1.05)
    else:
        ax.legend(loc="best")
    ax.grid(True)

    df = pd.DataFrame({'fprs':fprs, 'tprs':tprs, 'threshs':threshs, 'thresh': thresh, 'p_thresh':roc_metrics.p_thresh})
    if csv_save_path is not None:
        df.to_csv(str(csv_save_path), index=False)

    if show:
        plt.tight_layout()
        plt.show()

    if fig and fig_save_path is not None:
        plt.savefig(str(fig_save_path), bbox_inches='tight', dpi=300)
    if fig and show:
        plt.close(fig)

    return df

def plot_roc_curves_df_cols(df, target, predictors=None, ax=None, show=False, predictor_label_map={},
                            title=None, fig_save_path=None, figsize=None, dropna=False, **kwargs):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    if predictors is None:
        predictors = [str(predic) for predic in df.columns if str(predic)!=target]
    elif len(predictors)==0: raise ValueError('no predictors given, also predictors not set to None')
    results_df = []
    df_metrics = []
    for predictor in predictors:
        if 'tils' in predictor.lower() and predictor not in df:
            print('ignore missing predictor %s' % predictor)
            continue
        predictor_label = predictor_label_map.get(predictor, predictor)
        dfp = df
        if dropna:
            dfp = df[[target, predictor]].dropna()
        print(f'plotting roc for {predictor}->{target}, {len(dfp)} samples, {dfp[target].sum()} {target}')
        df_result = plot_roc_curve(dfp[target], dfp[predictor], ax=ax, label=predictor_label, title=title, **kwargs)
        df_result['predictor'] = predictor
        results_df.append(df_result)
        # metrics['name'] = predictor
        # metrics.pop('pred',None)
        # metrics.pop('tprs',None)
        # metrics.pop('fprs',None)
        # metrics.pop('threshs',None)
        # df_metrics.append(metrics)
    # df_metrics = pd.DataFrame(df_metrics)
    df_results = pd.concat(results_df, ignore_index=True, axis=0)
    if fig is not None and fig_save_path is not None:
        legs = [c for c in ax.get_children() if isinstance(c, matplotlib.legend.Legend)]
        plt.savefig(str(fig_save_path), bbox_inches='tight', dpi=300, legs=legs)
    if show:
        plt.tight_layout()
        plt.show()
    # return df_results, df_metrics
    return df_results

def plot_roc_curves_df(df, target_col, predictor_col, prob_col, ax=None, show=False, predictor_label_map={},
                       title=None, fig_save_path=None, **kwargs):
    #TODO: make class roc-analysis
    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    predictors = list(df[predictor_col].unique())
    results_df = []
    for predictor in predictors:
        df_pr = df[df[predictor_col]==predictor]
        predictor_label = predictor_label_map.get(predictor, predictor)
        df_result, metrics = plot_roc_curve(df_pr[target_col], df_pr[prob_col], ax=ax, label=predictor_label, title=title, **kwargs)
        df_result['predictor'] = predictor
        results_df.append(df_result)

    df_results = pd.concat(results_df, ignore_index=True, axis=0)
    if fig is not None and fig_save_path is not None:
        legs = [c for c in ax.get_children() if isinstance(c, matplotlib.legend.Legend)]
        plt.savefig(str(fig_save_path), bbox_inches='tight', dpi=300, legs=legs)
    if show:
        plt.tight_layout()
        plt.show()
    return df_results

def _map_labels(y_true, label_map):
    if label_map is not None and len(label_map)>0:
        mapped = [label_map[y] for y in y_true]
        y_true = mapped
    return y_true

def roc_auc_df(df, target, predictors):
    results = []
    for predictor in predictors:
        auc = roc_auc(df[target], probs=df[predictor])
        results.append(dict(var=predictor, auc=auc))
    df = pd.DataFrame(results)
    return df

def clf_metrics(targ, pred=None, probs=None, target_label_map=None, detail=True,
                save_plots=True, out_dir=None, class_names=None,
                with_cm=False, auc_ptest=False, show=False, silent=False, kappa_weights=None,
                ret_pred=False, multilabel=None, cm_xlabel='Predicted', cm_ylabel='True'):
    """
    targ: labels or one-hot, one-hot for multilabel
    pred: labels or one-hot, one-hot for multilabel
    probs: column for binary clf, array (n_samples, n_classes) for multilabel
    """
    if out_dir is None:
        save_plots = False
    else:
        out_dir = Path(out_dir)
        ensure_dir_exists(out_dir)

    if save_plots and out_dir is not None:
        roc_path = out_dir/'roc.jpg'
        roc_csv_path = out_dir/'roc.csv'
        roc_neg_path = out_dir/'roc_neg.jpg'
        roc_neg_csv_path = out_dir/'roc_neg.csv'
        cm_path = out_dir/'cm.jpg'
        out_path = out_dir/'metrics.json'

    #determine multilabel from pred and target if not given explictly
    #(can be wrong if in the given data there are no multi-targets or -predictions
    #(but then the metrics should be the same as if it was multiclass anyway)
    if multilabel is None:
        if len(targ.shape)==1 or targ.shape[1]==1:
            multilabel = False
        else:
            #target one-hot encoded
            utarg = list(set(targ.sum(axis=1)))
            if len(utarg)==1 and 1 in utarg:
                targ_multilabel = False
            else:
                targ_multilabel = True

            if pred is not None:
                upred = list(set(pred.sum(axis=1)))
                if len(upred)==1 and 1 in upred:
                    pred_multialbel = False
                else:
                    pred_multialbel = True
                multilabel = targ_multilabel or pred_multialbel
            else:
                multilabel = targ_multilabel

    #if multilabel is false, but hot-encoded target -> argmax
    if not multilabel and len(targ.shape)==2 and targ.shape[1]>1:
        targ = targ.argmax(1)[:,None]

    if multilabel and (len(targ.shape)!=2 or targ.shape[1]<=1):
        raise ValueError('multilabel flag, but target shape %s' % str(targ.shape))
    # multilabel = False
    # if len(targ.shape)==2 and targ.shape[1]>1: #multilabel, hot encoding
    if multilabel: #multilabel, hot encoding
        n_classes = max(2, targ.shape[1])
        # multilabel = True
    else:#binary or multiclass, labels directly
        # n_classes = max(len(np.unique(targ)), len(np.unique(pred)))
        if class_names is None:
            n_classes = len(np.unique(targ))
        else:
            n_classes = len(class_names)

    if pred is None and probs is None:
        raise ValueError('either pred or probs must be not None')

    if pred is None:
        if multilabel:
            pred = (probs>=0.5).astype(np.uint8)
        elif len(probs.shape)==2 and probs.shape[1]>1:
            #multiclass (or binary with two-dim probs
            pred = np.argmax(probs, axis=1)
        else: #binary clf
            assert len(probs.shape)==1 or (len(probs.shape)==1 and probs.shape[1]==1),\
                'probs.shape %s incompatible with binary clf' % str(probs.shape)
            pred = (probs.flatten()>=0.5).astype(np.uint8)


    if target_label_map is not None and multilabel:
        raise ValueError('implement target label mapping when multilabel!')

    if (pred==0).all():
        print('Warning: all predictions are 0!')
    elif (pred==1).all():
        print('Warning: all predictions are 1!')

    try:
        targ = _map_labels(targ, target_label_map)
        acc = sklearn.metrics.accuracy_score(targ, pred)
    except:
        print(f'accuarcy computation failed, n_classes={n_classes}, multilabel={multilabel}',
              f'targ.shape:{targ.shape}, pred.shape:{pred.shape}')
        print('target:', targ)
        print('pred:', pred)
        raise
    results = {'acc':acc, 'n':len(targ)}

    if not multilabel:
        results['kappa'] = cohen_kappa_score(targ, pred, weights=kappa_weights)

    if n_classes<=2:
        f1 = sklearn.metrics.f1_score(targ, pred)
        results['f1'] = f1
    else:
        for avg in  ['micro', 'macro', 'weighted']:
            f1 = sklearn.metrics.f1_score(targ, pred, average=avg)
            results['f1_'+avg] = f1

    if probs is not None:
        if isinstance(probs, (list, tuple)):
            probs = np.array(probs)
        if len(probs.shape)==1 or probs.shape[1]==2:
            results['auc'] = roc_auc(targ, probs)
            if auc_ptest:
                from wsilearn.utils.permutation_test import roc_auc_permutation_test
                auc_p = roc_auc_permutation_test(targ, probs)
                results['auc_p'] = auc_p
            if save_plots:
                pos_label=np.max(targ) #should be 1 always
                plot_roc_curve(targ, probs, pos_label=pos_label, show=show,
                               fig_save_path=roc_path, csv_save_path=roc_csv_path, silent=silent)
                plot_roc_curve(pos_label-targ, pos_label-probs, pos_label=pos_label, show=show,
                               fig_save_path=roc_neg_path, csv_save_path=roc_neg_csv_path, silent=silent,
                               xlabel='False Negative Rate', ylabel='True Negative Rate',)
        elif n_classes>2:
            aucs = []
            if multilabel:
                binary_labels = targ
            else:
                binary_labels = label_binarize(targ, classes=[i for i in range(n_classes)])
            for class_idx in range(n_classes):
                if np.sum(binary_labels[class_idx])>0:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], probs[:, class_idx])
                    auci = sklearn.metrics.auc(fpr, tpr)
                    aucs.append(auci)
                    if class_names is not None:
                        results[f'auc_{class_names[class_idx]}'] = auci
                    else:
                        results[f'auc_{class_idx}'] = auci
                    if save_plots:
                        plot_roc_curve_multi_one(binary_labels, probs, fig_save_path=roc_path,
                                             class_names=class_names, show=show, silent=silent)
                else:
                    #there is no ground truth with this label
                    aucs.append(float('nan'))
            auc = np.nanmean(np.array(aucs))
            results['auc'] = auc

    if n_classes==2 and not multilabel:
        results.update(binary_clf_pred_metrics(targ, pred, detail=detail, kappa_weights=kappa_weights))

    if multilabel:
        cm = multilabel_confusion_matrix(targ, pred)
    else:
        cm = confusion_matrix(targ, pred, labels=range(n_classes))
        cm_metrics = pycm_metrics(cm, class_names)
        metric_diffs = {k:v for k,v in results.items() if k in cm_metrics and not is_iterable(v) and str(v)[:8]!=str(cm_metrics[k])[:8]}
        if len(metric_diffs)!=0:
            cm_diffs = {k:v for k,v in cm_metrics.items() if k in metric_diffs}
            print('WARNING: result metrics different from cm_metrics, result values:', metric_diffs, 'cm values:', cm_diffs)
        for k,v in cm_metrics.items():
            if k not in results:
                results[k] = v
        # results.update(cm_metrics)

    if with_cm:
        results['cm'] = cm
    else:
        results.pop('cm',None)

    if save_plots and out_dir is not None:
        if class_names is None:
            class_names = [str(i) for i in range(n_classes)]
        parent_dir = Path(cm_path).parent
        ensure_dir_exists(parent_dir)
        xticklabel_rotation = 45
        if np.max([len(s) for s in class_names])<=5:
            xticklabel_rotation = 0
        try:
            if multilabel:
                #we assume the class importance to increase with the class index and evaluate
                #the prediction as multiclass taking only the class with the highest index in consideration
                y_true_mc = n_classes - np.argmax(targ[:, ::-1], 1) - 1
                pred_mc = n_classes-np.argmax(pred[:,::-1], 1)-1
                cm_max = confusion_matrix(y_true_mc, pred_mc, labels=range(n_classes))
                max_metrics = pycm_metrics(cm_max, class_names)
                # max_metrics = {k+'_max':v for k,v in max_metrics.items()}
                results['multimax'] = max_metrics
                save_path_max = parent_dir / (Path(cm_path).stem + '_max' + Path(cm_path).suffix)
                save_path_max_abs = parent_dir / (Path(cm_path).stem + '_max_abs' + Path(cm_path).suffix)
                plot_confusion_matrix(cm_max, class_names, normalize=True, save_path=save_path_max,
                            xticklabel_rotation=xticklabel_rotation)
                plot_confusion_matrix(cm_max, class_names, normalize=False, save_path=save_path_max_abs,
                            xticklabel_rotation=xticklabel_rotation)

                cm_plot_fct = plot_multilabel_confusion_matrix
            else:
                cm_plot_fct = plot_confusion_matrix
            cm_plot_fct(cm, class_names, normalize=True, save_path=cm_path,
                        xticklabel_rotation=xticklabel_rotation, xlabel=cm_xlabel, ylabel=cm_ylabel)
            abs_save_path = parent_dir / (Path(cm_path).stem + '_abs' + Path(cm_path).suffix)
            cm_plot_fct(cm, class_names, normalize=False, save_path=abs_save_path,
                                      xticklabel_rotation=xticklabel_rotation, xlabel=cm_xlabel, ylabel=cm_ylabel)
        except:
            print('Exception when computing confusion matrix')
            traceback.print_exc()
            # print(sys.exc_info())

    if out_dir is not None:
        write_json_dict(path=out_path, data=results)
    if ret_pred:
        return results, pred
    return results

def clf_metrics2(targ, probs, class_names, thresh_tpr=None, tpr_weight=None, detail=True,
                 save_plots=False, out_dir=None,
                 show=False, silent=False, kappa_weights=None,
                 ret_pred=False, cm_xlabel='Predicted', cm_ylabel='True', ret_ax=False):
    """
    targ: labels e.g. [0, 2, 1] or one-hot for multilabel, e.g. [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
    probs: array (n_samples, n_classes) for multiclass or multilabel, also allows list or column for binary clf
    """
    if save_plots and out_dir is None: raise ValueError('save_plots but no out_dir!')
    if save_plots:
        out_dir = Path(out_dir)
        ensure_dir_exists(out_dir)
        roc_path = out_dir/'roc.jpg'
        roc_csv_path = out_dir/'roc.csv'
        roc_neg_path = out_dir/'roc_neg.jpg'
        roc_neg_csv_path = out_dir/'roc_neg.csv'
        cm_path = out_dir/'cm.jpg'
        out_path = out_dir/'metrics.json'

    # targ = _map_labels(targ, target_label_map)
    targ = np.array(targ).squeeze()
    probs = np.array(probs).squeeze()

    n_classes = len(class_names)
    multilabel = len(targ.shape)>1
    multiclass = n_classes > 2
    binary = n_classes == 2 and not multilabel

    if binary and len(probs.shape)==2 and probs.shape[1]==2:
        probs = probs[:,1]
    if binary and thresh_tpr is not None and is_iterable(thresh_tpr) and len(thresh_tpr)==2:
        if len(thresh_tpr)>2: raise ValueError("thresh_tpr with more then 2 entries: %s" % str(thresh_tpr))
        thresh_tpr = thresh_tpr[1]

    results = {}
    if binary:
        #binary
        roc_results = roc_analysis(targ, probs, thresh_tpr=thresh_tpr, tpr_weight=tpr_weight, detail=False)
        thresh_tpr = roc_results['thresh']
        results.update(roc_results)
        pred = roc_results.pop('pred')
        if save_plots:
            plot_roc_curve(targ, probs, thresh_tpr=thresh_tpr, show=show,
                           fig_save_path=roc_path, csv_save_path=roc_csv_path, silent=silent)
            plot_roc_curve(1-targ, 1-probs, thresh_tpr=1-thresh_tpr, show=show,
                           fig_save_path=roc_neg_path, csv_save_path=roc_neg_csv_path, silent=silent,
                           xlabel='False Negative Rate', ylabel='True Negative Rate',)

    if not is_iterable(thresh_tpr):
        thresh_tpr = [thresh_tpr] * n_classes
    if not is_iterable(tpr_weight):
        tpr_weight = [tpr_weight]*n_classes

    #compure roc-metrics and predicted labels
    ax_multiroc = None
    if multiclass or multilabel:
        aucs = []
        if multilabel:
            binary_labels = targ
            pred = np.zeros_like(targ)
        else: #multiclass
            pred = np.argmax(probs, axis=1)
            binary_labels = label_binarize(targ, classes=[i for i in range(n_classes)])
        #compute aucs: one class vs rest
        for class_idx in range(n_classes):
            if np.sum(binary_labels[:,class_idx])>0:
                roc_results_i = roc_analysis(binary_labels[:, class_idx], probs[:, class_idx], kappa_weights=kappa_weights,
                                             thresh_tpr=thresh_tpr[class_idx], tpr_weight=tpr_weight[class_idx])
                thresh_tpr[class_idx] = roc_results_i['thresh']
                aucs.append(roc_results_i['auc'])
                if multilabel:
                    pred[:,class_idx] = roc_results_i['pred']
                # for m in ['auc','acc','f1']:
                for m in ['auc','thresh']:
                    results[f'{m}_{class_names[class_idx]}'] = roc_results_i[m]
            else:
                #there is no ground truth with this label
                aucs.append(float('nan'))
        if save_plots:
            fig_multiroc, ax_multiroc = plot_roc_curve_multi_one(binary_labels, probs, fig_save_path=roc_path, thresh_tpr=thresh_tpr, tpr_weight=tpr_weight,
                                         class_names=class_names, show=show, silent=silent)

        auc = np.nanmean(np.array(aucs))
        results['auc'] = auc

    pred = pred.astype(np.uint8)
    if (pred==0).all():
        print('Warning: all predictions are 0!')
    elif (pred==1).all():
        print('Warning: all predictions are 1!')

    if detail:
        results['pred'] = pred
    # if pred is None:
    #     if multilabel:
    #         pred = (probs>=0.5).astype(np.uint8)
    #     elif len(probs.shape)==2 and probs.shape[1]>1:
    #         #multiclass (or binary with two-dim probs
    #         pred = np.argmax(probs, axis=1)
    #     else: #binary clf
    #         assert len(probs.shape)==1 or (len(probs.shape)==1 and probs.shape[1]==1),\
    #             'probs.shape %s incompatible with binary clf' % str(probs.shape)
    #         pred = (probs.flatten()>=0.5).astype(np.uint8)

    acc = sklearn.metrics.accuracy_score(targ, pred)
    results['acc'] = acc
    results['n'] = len(targ)
    if not multilabel:
        results['kappa'] = cohen_kappa_score(targ, pred, weights=kappa_weights)

    if n_classes<=2:
        if multilabel:
            f1 = sklearn.metrics.f1_score(targ, pred, average='macro')
        else:
            f1 = sklearn.metrics.f1_score(targ, pred)
        results['f1'] = f1
    else:
        for avg in  ['micro', 'macro', 'weighted']:
            f1 = sklearn.metrics.f1_score(targ, pred, average=avg)
            results['f1_'+avg] = f1

    if multilabel:
        cm = multilabel_confusion_matrix(targ, pred)
    else:
        cm = confusion_matrix(targ, pred, labels=range(n_classes))
        cm_metrics = pycm_metrics(cm, class_names)
        metric_diffs = {k:v for k,v in results.items() if k in cm_metrics and not is_iterable(v) and str(v)[:8]!=str(cm_metrics[k])[:8]}
        if len(metric_diffs)!=0:
            cm_diffs = {k:v for k,v in cm_metrics.items() if k in metric_diffs and k!='kappa'}
            print('WARNING: result metrics different from cm_metrics, result values:', metric_diffs, 'cm values:', cm_diffs)
        results.update(cm_metrics)

    results['cm'] = cm


    if save_plots:
        parent_dir = Path(cm_path).parent
        ensure_dir_exists(parent_dir)
        xticklabel_rotation = 45
        if np.max([len(s) for s in class_names])<=5:
            xticklabel_rotation = 0
        try:
            if multilabel:
                #we assume the class importance to increase with the class index and evaluate
                #the prediction as multiclass taking only the class with the highest index in consideration
                y_true_mc = n_classes - np.argmax(targ[:, ::-1], 1) - 1
                pred_mc = n_classes-np.argmax(pred[:,::-1], 1)-1
                cm_max = confusion_matrix(y_true_mc, pred_mc, labels=range(n_classes))
                max_metrics = pycm_metrics(cm_max, class_names)
                results['multimax'] = max_metrics
                save_path_max = parent_dir / (Path(cm_path).stem + '_max' + Path(cm_path).suffix)
                save_path_max_abs = parent_dir / (Path(cm_path).stem + '_max_abs' + Path(cm_path).suffix)
                plot_confusion_matrix(cm_max, class_names, normalize=True, save_path=save_path_max,
                            xticklabel_rotation=xticklabel_rotation)
                plot_confusion_matrix(cm_max, class_names, normalize=False, save_path=save_path_max_abs,
                            xticklabel_rotation=xticklabel_rotation)

                cm_plot_fct = plot_multilabel_confusion_matrix
            else:
                cm_plot_fct = plot_confusion_matrix
            cm_plot_fct(cm, class_names, normalize=True, save_path=cm_path,
                        xticklabel_rotation=xticklabel_rotation, xlabel=cm_xlabel, ylabel=cm_ylabel)
            abs_save_path = parent_dir / (Path(cm_path).stem + '_abs' + Path(cm_path).suffix)
            cm_plot_fct(cm, class_names, normalize=False, save_path=abs_save_path,
                                      xticklabel_rotation=xticklabel_rotation, xlabel=cm_xlabel, ylabel=cm_ylabel)
        except:
            print('Exception when computing confusion matrix')
            traceback.print_exc()

    if out_dir is not None:
        write_json_dict(path=out_path, data=results)
    if ret_ax:
        results['ax_multiroc'] = ax_multiroc
        results['fig_multiroc'] = fig_multiroc
    if ret_pred:
        return results, pred
    return results

def save_cm_as_csv(cm, classes, path, yclasses=None):
    if yclasses is None:
        yclasses = classes
    df = pd.DataFrame(cm, columns=yclasses, index=classes)
    df.to_csv(str(path))


def rearrange_cm(cm, class_names, target_class_names):
    new_cm = np.zeros_like(cm)
    for i,cli in enumerate(target_class_names):
        si = class_names.index(cli)
        for j,clj in enumerate(target_class_names):
            sj = class_names.index(clj)
            new_cm[i,j] = cm[si,sj]
    return new_cm


def plot_cooccurrence_df(df, vars=None, **kwargs):
    if vars is None:
        vars = list(df.columns.values)
    df = df[vars]
    df_coocc = df.T.dot(df)
    if len(vars)>10:
        kwargs['font_size'] =7
    plot_cooccurrence_matrix_df(df_coocc, **kwargs)

def plot_single_occurrence_df(df, vars=None, **kwargs):
    if vars is None:
        vars = list(df.columns.values)
    df = df[vars]
    df = df[df[vars].sum(1)<2]
    plot_cooccurrence_df(df, vars, **kwargs)

def plot_cooccurrence_matrix_df(df, vars=None, title=None, **kwargs):
    """
    df: square (diagonal-symmetric) matrix with cooccurrences.
    """
    cm = df.values
    if title is None:
        title = 'Co-occurrence matrix'
    if vars is None:
        vars = list(df.columns.values)
    else:
        cm = rearrange_cm(cm, list(df.columns.values), vars)
    return plot_cooccurrence_matrix(cm, vars, title=title, **kwargs)


def adjust_ax(ax, xticklabel_rotation=0, hide_xticks=False, hide_yticks=False, tick_size=None, title=None):
    if xticklabel_rotation !=0 and not hide_xticks:
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=xticklabel_rotation, ha="right",
                 rotation_mode="anchor")

    if hide_xticks:
        ax.get_xaxis().set_visible(False)
    if hide_yticks:
        ax.get_yaxis().set_visible(False)

    if tick_size is not None:
        ax.tick_params(axis='both', which='major', labelsize=tick_size)
        ax.tick_params(axis='both', which='minor', labelsize=tick_size)

    if title is not None:
        ax.set_title(title)

def plot_cooccurrence_matrix(cm, xlabels, ylabels=None, normalize=False, title=None, cmap=plt.cm.Blues, vmin=None, vmax=None,
                             verbose=False, xticklabel_rotation=45, show=False, max_class_text=30, close_fig=True,
                             ax=None, xlabel=None, ylabel=None, colorbar=False, font_size = None, fmt_norm = '.2f',
                             hide_yticks=False, hide_xticks=False, save_path=None, mask=None, block=True, figsize=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if ylabels is None:
        ylabels = xlabels
    n_classes = len(xlabels)
    # yticklabels = classes

    cm = np.nan_to_num(cm)

    if 'float' in str(cm.dtype) and ((cm % 1)==0).all():
        cm = cm.astype(np.int)

    if matplotlib.__version__ == '3.1.0' or matplotlib.__version__ == '3.1.1':
        #fixes a bug cutting off half of the top and bottom row; should be fixed in 3.1.2
        ax.set_ylim(-0.5, len(cm)-0.5)
        cm = np.flip(cm, axis=0)
        ylabels = ylabels[::-1]
    else:
        try:
            version_as_number = int(matplotlib.__version__.replace('.', ''))
            if version_as_number > 311 and version_as_number < 330: #seems to work for 330
                print('New matplotlib version: %d > 3.1.1. Check whether confusion matrix plotting is still/again valid!' % version_as_number)
        except Exception as ex:
            print('unknown matplotlib version formatting! cant check whether confusion matrix plotting will be valid')

    if np.max(cm)>1:
        cm_rel = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        # already normalized
        cm_rel = cm.copy()
        normalize = True

    cm_rel = np.nan_to_num(cm_rel)
    # print(cm)

    if ax is None:
        if n_classes > 8:
            size_add = int(np.sqrt(n_classes))
            if figsize is None:
                figsize = figsize=(5+size_add,5+size_add)
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1, 1, 1)
        else:
            if figsize is None:
                figsize = (5,5)
            fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None


    # im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    cm_show = cm_rel
    if mask is not None:
        cm_show = np.ma.masked_where(mask!=1, cm_show)
    im = ax.imshow(cm_show, interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
    if colorbar:
        ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=xlabels if not hide_xticks else [], yticklabels=ylabels,
           title=title, ylabel=ylabel, xlabel=xlabel)

    if xticklabel_rotation !=0 and not hide_xticks:
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=xticklabel_rotation, ha="right",
                 rotation_mode="anchor")

    if hide_xticks:
        ax.get_xaxis().set_visible(False)
    if hide_yticks:
        ax.get_yaxis().set_visible(False)

    # Loop over data dimensions and create text annotations.
    if font_size is None:
        font_size = matplotlib.rcParams['font.size']
    if not normalize:
        #adapt font size for verly large counts
        if cm.max()>100000:
            font_size = int(font_size - font_size/3)
        elif cm.max()>10000:
            font_size = int(font_size - font_size/5)
        font_size = max(5, font_size)

    font_size+=2
    ax.xaxis.set_tick_params(labelsize=font_size)
    ax.yaxis.set_tick_params(labelsize=font_size)

    ax.xaxis.label.set_size(font_size)
    ax.yaxis.label.set_size(font_size)
    ax.title.set_size(font_size)

    cm_sel = cm_rel
    fmt_abs = 'd'
    cmap_min, cmap_max = cmap(0), cmap(256)
    if n_classes <= max_class_text:
        # fmt = fmt_norm if normalize else fmt_abs
        # thresh = cm.max() / 2.
        threshp = max(0, cm_sel.min()) + (max(0,cm_sel.max()) - max(0,cm_sel.min())) / 2.
        threshn = min(0,cm_sel.min()) + (min(0,cm_sel.max()) - min(0,cm_sel.min())) / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                text_color = 'white' if cm_sel[i, j] > threshp or cm_sel[i, j] < threshn else 'black'
                if normalize:
                    text_val = format(cm_rel[i, j], fmt_norm)
                else:
                    text_val = format(cm[i, j], fmt_abs)
                # text_val =  format(cm[i, j], fmt)
                ax.text(j, i, text_val,
                        ha="center", va="center", fontsize=font_size, color=text_color
                        # color=cmap_min if cm[i, j] > thresh else cmap_max
                        # color='white' if cm[i, j] > thresh else 'black'
                        )

    if show:
        plt.tight_layout()
        plt.show(block=block)

    if save_path is not None:
        # fig.savefig(str(save_path))
        mkdir(Path(save_path).parent)
        fig.tight_layout()
        fig.savefig(str(save_path), bbox_inches="tight", dpi=300)
        if verbose: print('%s saved' % save_path)

    if close_fig and fig is not None:
        plt.close(fig)
    else:
        return fig, ax

def plot_multilabel_confusion_matrix(cm, classes, normalize=False, title=None, cmap=plt.cm.Blues,
                          save_path=None, save_csv_path=None, xlabel='Predicted', ylabel='True',
                          verbose=False, show=False, xticklabel_rotation=0, **kwargs):
    """
    cm has the shape (n_classes, 2, 2) from sklearn
    """
    if title is None:
        title = 'Confusion Matrix'
    cm = np.nan_to_num(cm)
    if len(cm)>5:
        ncols = max(3, math.ceil(math.sqrt(len(cm))))
        nrows = math.ceil(len(cm)/ncols)
    else:
        ncols = len(cm)
        nrows = 1
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3*ncols, 3))
    axs = axs.ravel()
    fig.suptitle(title)

    if save_path is not None:
        save_path = Path(save_path)

    results = []
    for i in range(len(cm)):
        cmi = cm[i]
        # if save_csv_path == True and save_path is not None:
        #     save_path = Path(save_path)
        if save_csv_path is not None:
            save_csv_path = save_path.parent/(save_path.stem+'_%s.csv' % str(classes[i]))
            save_cm_as_csv(cmi, classes, save_csv_path)
        results.append(plot_cooccurrence_matrix(cmi, xlabels=['0', '1'], cmap=cmap, verbose=verbose,
                                                xticklabel_rotation=0, show=False, ax=axs[i], xlabel=xlabel, ylabel=ylabel,
                                                title=classes[i], normalize=normalize, **kwargs))

    if save_path is not None:
        fig.tight_layout()
        fig.savefig(str(save_path), bbox_inches="tight")
        if verbose: print('%s saved' % save_path)

    if show:
        plt.tight_layout()
        plt.show()
    else:
        plt.close(fig)

def plot_confusion_matrix(cm, classes, ylabels=None, normalize=False, title=None, cmap=plt.cm.Blues,
                          save_path=None, save_csv_path=None, xlabel='Predicted', ylabel='True',
                          verbose=False, xticklabel_rotation=45, show=False, ax=None, **kwargs):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if title is None:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix'

    cm = np.nan_to_num(cm)

    if save_csv_path == True and save_path is not None:
        save_path = Path(save_path)
        save_csv_path = save_path.parent/(save_path.stem+'.csv')
    if save_csv_path is not None:
        save_cm_as_csv(cm, classes, save_csv_path, yclasses=ylabels)

    return plot_cooccurrence_matrix(cm, classes, ylabels=ylabels, cmap=cmap, verbose=verbose, xticklabel_rotation=xticklabel_rotation, show=show,
                                    ax=ax, xlabel=xlabel, ylabel=ylabel, title=title, save_path=save_path, normalize=normalize, **kwargs)

def _test_clf_metrics_multi(**kwargs):
    y = np.array([[1, 1, 1],[1,1,0],[1,0,1],[0, 1, 1],[0, 1, 0], [1, 0, 0]])
    probs = np.array([[.2, .2, .6],[.3, .4, .3],[.5, .2, .3],[.2, .4, .4], [.1, .4, .5],[.6, .2, .2]])
    preds = np.array([[0, 0, 1],   [0,1,0],     [1,0,0],     [0, 1, 1],    [0, 1, 1],   [1, 0, 0]])
    clf_metrics(y, preds, probs, out_dir='./out', **kwargs)


if __name__ == '__main__':
    # _test_clf_metrics_multi(show=True)
    pass


class PredictionsDf(object):
    """
    Probabilities: binary clf: prob
        multiclass/multilabel: prob0, prob1, ...
    Predictions binary and multiclass clf: pred (prediction label)
        multilabel: target: list of labels e.g. 1,2
    Predictions: regr: pred, regr with multiple targets: pred0, pred1, ...

    Targets: regr: target, regr with multiple targets: target0, target1, ...
    Targets: binary and multiclass: target (target label)
        multilabel: target: list of labels e.g. 1,2; hot-encoding target0, target1, ...

    """
    name_col = 'name'
    target_col = 'target'
    pred_col = 'pred'
    prob_col = 'prob'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, arg):
        return self.df[arg]

    def __setitem__(self, key, value):
        self.df[key] = value

    def __contains__(self, key):
        return self.has_col(key)

    def has_col(self, col):
        return col in self.df.columns

    @staticmethod
    def get_prob_col_name(i=None):
        if i is None:
            return PredictionsDf.prob_col
        else:
            return PredictionsDf.prob_col+str(i)

    @staticmethod
    def get_target_col_name(i=None):
        if i is None:
            return PredictionsDf.target_col
        else:
            return PredictionsDf.target_col+str(i)

    @staticmethod
    def get_pred_col_name(i=None):
        if i is None:
            return PredictionsDf.pred_col
        else:
            return PredictionsDf.pred_col+str(i)

    def __init__(self, data=None, sort_columns=False):
        self.df = None
        if data is not None:
            if is_df(data):
                self.df = data
            else:
                self.df = pd.DataFrame(data)
            if sort_columns:
                self.df = self.df[sorted(self.df.columns)]
            self._init()

    def _init(self):
        out_cols = self._prob_cols()
        if len(out_cols)==0:
            out_cols = self._pred_cols()
        self.n_targets = len(out_cols) #output dimensions

        if len(self._prob_cols()):
            targ_hot = np.zeros((len(self.df), self.n_targets), dtype=np.uint8)
            for row, targ in enumerate(self.df[self.target_col]):
                targ = parse_string(targ)
                if not is_iterable(targ):
                    targ = [targ]
                for col in range(self.n_targets):
                    if col in targ:
                        targ_hot[row,col] = 1
            for col in range(self.n_targets):
                tcol = PredictionsDf.get_target_col_name(col)
                if tcol in self.df:
                    if not (np.array(self.df[tcol])==targ_hot[:,col]).all():
                        raise ValueError('Error in csv-target-encoding or BUG! different targets for column %s: \n%s and \n%s' %\
                            (tcol, str(np.array(self.df[tcol])), str(targ_hot[:,col])))
                else:
                    self.df[tcol] = targ_hot[:,col]

    def save(self, path):
        self.df.to_csv(str(path), index=None)

    @staticmethod
    def from_csv(path):
        pdf = PredictionsDf()
        df = pd.read_csv(str(path))
        pdf.df = df
        pdf._init()
        return pdf

    def _prob_cols(self):
        return [str(c) for c in self.df.columns if self.prob_col in c]
    def _pred_cols(self):
        return [str(c) for c in self.df.columns if self.pred_col in c]

    def get_probs(self):
        prob_cols = self._prob_cols()
        dfp = self.df[prob_cols]
        return dfp.values

    def get_preds(self):
        pred_cols = self._pred_cols()
        if len(pred_cols)==0:
            return None
        dfp = self.df[pred_cols]
        return dfp.values

    def get_pred_only_df(self):
        cols = self._prob_cols()+self._pred_cols()
        cols.insert(0, self.name_col)
        return self.df[cols]

    def get_target(self):
        try:
            targets = self.df[self.target_col].values
            if len(targets)>0:
                if is_string(targets[0]):
                    targets = [parse_string(t) for t in targets]
                if is_iterable(targets[0]):
                    lens = [len(t) for t in targets]
                    if (np.array(lens)==1).all():
                        targets = [t[0] for t in targets]
            targets = np.array(targets)
            return targets
        except:
            print('Failed getting targets from column %s' % self.target_col)
            print_df(self.df)
            raise

    def get_target_hot(self, i=None):
        if i is None:
            tar_cols = [self.target_col +f'{i}' for i in range(self.n_targets)]
        else:
            tar_cols = [self.target_col+str(i)]
        return self.df[tar_cols].values

    def get_prob(self, i=''):
        return self.df[self.prob_col+i]

    def find_thresholds(self):
        prob_cols = self._prob_cols()
        threshs = []
        for i,pcol in enumerate(prob_cols):
            assert str(i) in pcol
            tar_col = self.get_target_hot(i)
            resulti = roc_analysis(tar_col, self[pcol], detail=False)
            threshs.append(resulti['thresh'])
        return np.array(threshs)

    def head(self, *args, **kwargs):
        return self.df.head(*args, **kwargs)

    def print(self, *args, **kwargs):
        print_df(self.df, *args, **kwargs)

def create_short_info_file(out_dir, result_metrics, purposes={'testing':'test', 'validation':'val'}):
    """ creates an empty file putting some key-values in the name. This is just to see the result of an experiment
    immediately when opening the result-directory without first to have to open some file.
    """
    short_name = ''
    name_result_map = OrderedDict()
    for purp, purp_short in purposes.items():
        if purp in result_metrics:
            name_result_map[purp_short] = result_metrics[purp]

    if len(name_result_map) != 0:
        for name, short_dict in name_result_map.items():
            if len(short_name)>0: name = '_'+name
            short_name+=name
            if 'auc' in short_dict:
                short_name += '_auc%d' % int(100 * short_dict['auc'])
            elif 'acc' in short_dict:
                short_name += '_acc%d' % int(100 * short_dict['auc'])
            if 'f1_macro' in short_dict:
                short_name += '_f%d' % int(100 * short_dict['f1_macro'])
            if 'cind' in short_dict:
                short_name += '_cind%d' % int(100 * short_dict['cind'])
            # if 'kappa' in short_dict:
            #     short_name += '_kappa%d' % int(100 * short_dict['kappa'])
    if len(short_name)>0:
        short_info_path = Path(out_dir) / short_name
        short_info_path.touch()
    return short_name

def create_simple_short_info_file(out_dir, metrics:dict):
    infos = []
    for k in ['auc','acc','error','cind']:
        if k in metrics:
            infos.append(f'{k}{metrics[k]:.3f}')
    if len(infos)>0 and out_dir is not None:
        name = '_'.join(infos)
        ensure_dir_exists(out_dir)
        short_info_path = Path(out_dir) / name
        short_info_path.touch()

def _plot_roc_curve_multi_one_example():
    n_classes = 3
    set_seed(10)
    labels = np.random.randint(0, n_classes+1, size=100)
    # labels = labels[:,None]
    probs = np.random.rand(100,3)
    plot_roc_curve_multi_one(labels, probs, show=True)

def plot_calibration(probs, y):
    """ plots the calibration curve/reliability diagram:
    true positive rate in dependence to predicted probability
    brier score: lower value: better calibration (but not always)
    """
    clf_score = brier_score_loss(y, probs, pos_label=y.max())
    fig = plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    ax1.grid(True)
    fraction_of_positives, mean_predicted_value = calibration_curve(y, probs, n_bins=10)
    ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
             label="brier score %1.3f" % (clf_score))
    ax2.hist(probs, range=(0, 1), bins=10,  # label=name,
             histtype="step", lw=2)
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')
    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)
    plt.tight_layout()
    plt.show()


def _test_clf_metrics():
    #### binary
    # results = clf_metrics2([0, 1, 0, 1], probs=[0.1, 0.2, 0.6, 0.7], class_names=['Cl0','Cl1'],
    #                        out_dir='./out/clf_metrics/binary_dummy', save_plots=True)
    # assert results['thresh']>0.6, results['thresh']
    #due to the 0.6 for 0, it has to set the threshold>0.6 to get optimal point
    # print(results)

    #### multiclass
    # #fixme cm with max entry=1 recognized as relative
    # results = clf_metrics2([0, 1, 0, 2, 2], probs=[[0.8, 0.1, 0.1],
    #                                                [0.1, 0.8, 0.1],
    #                                                [0.3, 0.3, 0.7],
    #                                                [0.3, 0.2, 0.5],
    #                                                [0.3, 0.3, 0.4]
    #                                                ],
    #                        out_dir='./out/clf_metrics/multiclass_dummy', save_plots=True,
    #                        class_names=['0','1','2'])
    # print(results)

    ##### multilabel
    results = clf_metrics2(targ=[[1, 0, 0],
                                 [0, 1, 0],
                                 [1, 0, 1],
                                 [0, 1, 0],
                                 [0, 0, 1]
                                 ],
                            probs=[[0.8, 0.1, 0.1],
                                   [0.6, 0.8, 0.1],
                                   [0.3, 0.3, 0.7],
                                   [0.3, 0.2, 0.6],
                                   [0.3, 0.3, 0.65]
                                   ],
                           out_dir='./out/clf_metrics/multilabel_dummy', save_plots=True,
                           class_names=['Cl0','Cl1','Cl2'])
    print(results)

if __name__ == '__main__':
    _plot_roc_curve_multi_one_example()
    _test_clf_metrics()