import numpy as np
from pycm import ConfusionMatrix
from sklearn.metrics import confusion_matrix, cohen_kappa_score


def create_pycm(cm, class_names):
    matrix = {}
    for i,cl_true in enumerate(class_names):
        cl_matrix = {}
        matrix[cl_true] = cl_matrix
        for j,cl_pred in enumerate(class_names):
            cl_matrix[cl_pred] = int(cm[i,j])

    cm = ConfusionMatrix(matrix=matrix)
    return cm

def pycm_metrics(cm, class_names, weights=None):
    cm = create_pycm(cm, class_names)

    #
    class_metrics = {'F1':'f1', 'PPV':'precision', 'TPR':'tpr', 'TNR':'tnr'}
    overall_metrics={'Overall ACC':'acc', 'F1 Macro':'f1_macro', 'F1 Micro':'f1_micro', 'Kappa':'kappa', 'SOA2(Fleiss)':'fleiss'}

    if weights is None:
        weights = {cl:{cli:((class_names.index(cli)-class_names.index(cl))**2) for cli in class_names} for cl in class_names}
    kappaq = cm.weighted_kappa(weights)
    results = {'kappa_quadratic':kappaq}
    for pycm_metric_name,metric_name in overall_metrics.items():
        results[metric_name] = cm.overall_stat[pycm_metric_name]
    for pycm_metric_name,metric_name in class_metrics.items():
        clm_results = cm.class_stat[pycm_metric_name]
        for class_name,v in clm_results.items():
            results[f'{metric_name}_{class_name}'] = v
    return results

if __name__ == '__main__':
    targ = [0, 0, 1, 1, 2, 3]
    pred = [3, 0, 2, 1, 3, 2]
    cm = confusion_matrix(targ, pred, labels=sorted(np.unique(targ)))
    cm_metrics = pycm_metrics(cm, class_names='a,b,c,d'.split(','))
    kw = cohen_kappa_score(targ, pred, weights='quadratic')
    k = cohen_kappa_score(targ, pred)
    print(k,kw)
    print(cm_metrics['kappa'], cm_metrics['kappa_quadratic'])
    assert np.isclose(k, cm_metrics['kappa'])
    assert np.isclose(kw, cm_metrics['kappa_quadratic'])