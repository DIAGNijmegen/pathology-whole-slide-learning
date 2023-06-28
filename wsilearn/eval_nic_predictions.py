from pathlib import Path

from wsilearn.utils.cool_utils import read_json_dict
from wsilearn.dataconf import DataConf
from wsilearn.utils.eval_utils import ClfEvaluator
from wsilearn.train_nic import ClfEvaluationHeatmapWriter
import pandas as pd

#some additional evaluation code

def eval_heatmaps(out_dir, n_hm_correct_max=20, overwrite=False, no_heatmaps=False, eval_out_dir=None):
    npz_dir_name = 'out_npz'

    out_dir = Path(out_dir)
    args = read_json_dict(out_dir/'args.json')
    class_names = args['class_names']
    train_type = args['train_type']
    conf_path = out_dir/Path(args['data_config']).name
    conf = DataConf(conf_path, train_type=train_type, target_names=class_names)
    anno_dir = args.get('anno_dir',None)

    hm_writer = ClfEvaluationHeatmapWriter(npz_dir=None, anno_dir=anno_dir, hmsoft=False,
                                           n_hm_correct_max=n_hm_correct_max, overwrite=overwrite)
    evaluator = ClfEvaluator(class_names=conf.get_target_cols(), callbacks=[] if no_heatmaps else [hm_writer])
    # eval_heatmaps()

    splits = conf.get_splits()
    splits.remove('validation')
    splits.insert(0, 'validation')

    results = {}
    for purp in splits:
        purp_out_dir = out_dir/('eval_%s' % purp)
        print('%s eval out_dir: %s' % (purp, str(purp_out_dir)))
        dfp = pd.read_csv(str(purp_out_dir/'predictions.csv'))
        hm_writer.npz_dir = purp_out_dir/npz_dir_name
        data_conf = DataConf(dfp, train_type=conf.train_type, target_names=conf.get_target_cols())
        if eval_out_dir is not None:
            purp_out_dir = f'{eval_out_dir}/eval_{purp}'
        pmetrics = evaluator.evaluate_all(data_conf, out_dir=purp_out_dir)
        results[purp] = pmetrics
    print('Done')

def eval_nic_predictions(model_dir, pred_path, out_dir):
    args = read_json_dict(Path(model_dir)/'args.json')
    class_names = args['class_names']
    train_type = args['train_type']
    results = read_json_dict(Path(model_dir)/'results.json')
    clf_thresholds = results.get('validation',{}).get('clf_thresholds', None)
    # dfp = pd.read_csv(str(pred_dir/'predictions.csv'))
    dfp = pd.read_csv(pred_path)
    conf = DataConf(dfp, train_type=train_type, target_names=class_names)
    evaluator = ClfEvaluator(class_names=conf.get_target_cols(), clf_thresholds=clf_thresholds)
    pmetrics = evaluator.evaluate_all(conf, out_dir=out_dir)
    print(pmetrics)

