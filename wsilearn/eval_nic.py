from pathlib import Path

from wsilearn.utils.flexparse import FlexArgumentParser
from wsilearn.utils.cool_utils import read_json_dict, write_json_dict
from wsilearn.train_nic import NicLearner


def eval_nic(model_dir, data_config, preprocess_dir, out_dir, no_heatmaps=False, no_out=False, cache_dir=None, **kwargs):
    args = read_json_dict(Path(model_dir)/ 'args.json')
    args.update(dict(out_dir=model_dir, data_config=data_config, preprocess_dir=preprocess_dir,
                     eval_out_dir=out_dir, autoname=False, exp_name=None, no_heatmaps=no_heatmaps,
                     cache_dir=cache_dir, no_out=no_out, heatmap_purposes=['all']))
    args.update(kwargs)

    results = read_json_dict(Path(model_dir)/'results.json')
    clf_thresholds = results.get('validation',{}).get('clf_thresholds',None)
    print('clf thresholds:', clf_thresholds)
    write_json_dict(path=Path(out_dir)/'args.json', data=args)
    trainer = NicLearner(**args, copyconf=False)
    trainer.eval_config(clf_thresholds=clf_thresholds)

if __name__ == '__main__':
    parser = FlexArgumentParser()
    parser.add_argument('--preprocess_dir', type=str, required=True,  help='compressed directory')
    parser.add_argument('--data_config', type=str, help='data csv', required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=False)
    # parser.add_argument('--skip_compression', default=False, action='store_true')

    args = parser.parse_args()
    eval_nic(**args)
