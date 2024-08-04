from pathlib import Path


from collections import defaultdict

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import pandas as pd
from torch.utils.data import DataLoader


from wsilearn.utils.cool_utils import read_json_dict, mkdir, Timer, is_string_or_path
from wsilearn.dataconf import DataConf, TrainType
from wsilearn.dl.pl.pl_inference import find_pl_model_path, load_pl_state_dict, InferenceOutWriter, Inferencer
from wsilearn.compress.compressed_dataset import CompressedDataset
from wsilearn.nic_utils import create_nic_model, HeatmapWriter

from wsilearn.compress.compress import compress

from wsilearn.utils.cool_utils import save_arrays, mkdir, list_intersection, dict_agg_type
from wsilearn.dataconf import TrainType
from wsilearn.utils.df_utils import unique_one, df_concat, df_save, print_df
from wsilearn.utils.path_utils import PathUtils
from wsilearn.nic_utils import HeatmapWriter


def combine_out(eval_dirs, out_dir, overwrite=False):
    print('combining %s' % str(eval_dirs))
    name_pathes_map = defaultdict(list)
    for ev in eval_dirs:
        ev = Path(ev)
        outs = PathUtils.list_pathes(ev, ending='npz')
        if len(outs)==0:
            ev = Path(ev)/'out_npz'
            outs = PathUtils.list_pathes(ev, ending='npz')
        for out in outs:
            name_pathes_map[out.stem].append(str(out))
    lens = [len(v) for k,v in name_pathes_map.items()]
    lens_unique = np.unique(lens)
    if len(lens_unique)!=1 or lens_unique[0]<2:
        raise ValueError('invalid unique pathes', lens_unique)

    mkdir(Path(out_dir))
    for name, pathes in tqdm(name_pathes_map.items()):
        out_path = Path(out_dir)/(name+'.npz')
        if out_path.exists() and not overwrite:
            continue
        contents = [np.load(str(path)) for path in pathes]
        agg = dict_agg_type(contents, number_unique=True, string_unique=True, rest_unique=True,
                            only_common_keys=True, ignore_keys=['image','path'])
        keys = list(agg.keys())
        save_arrays(out_path, **agg)
    print('Done!')

def combine_predictions(eval_dirs, out_dir, overwrite=False, id='name'):
    fnames = ['out.csv', 'predictions.csv']

    for fname in fnames:
        ppathes = [Path(ev)/fname for ev in eval_dirs]
        out_path = Path(out_dir)/fname
        if out_path.exists() and not overwrite or not ppathes[0].exists():
            continue
        dfs = [pd.read_csv(ppath) for ppath in ppathes]
        # for i,df in enumerate(dfs):
        #     fold_name = eval_dirs[i].split('fold')[-1].split('/')[0]
        #     df['fold'] = fold_name
        df = df_concat(*dfs)

        cols = [str(c) for c in df.columns if c not in ['path','image']]
        out_cols = [c for c in cols if c.startswith('out') or c.startswith('pred')]
        agg = {c:(c,unique_one) for c in cols}
        agg.update({c:(c,np.mean) for c in out_cols})
        agg.update({c+'_std':(c,np.std) for c in out_cols})
        df = df.groupby(id).agg(**agg)
        assert len(df)==len(dfs[0])
        # print_df(df.head())
        df_save(df, out_path)
        print(str(out_path))


def create_combined_heatmaps(npz_dir, out_dir, train_type, wsi_dir=None, overwrite=False, no_links=False):
    # rec = dfp.to_dict('records')[0]
    hm_writer = HeatmapWriter(npz_dir=npz_dir, overwrite=overwrite, no_links=no_links, train_type=train_type)
    npzs = PathUtils.list_pathes(npz_dir, ending='npz')
    mkdir(out_dir)
    for npz in tqdm(npzs):
        kwargs = {}
        if wsi_dir is not None:
            kwargs['wsi_path'] = Path(wsi_dir)/(npz.stem+'.tif')
        hm_writer.save_hm_and_patches_with(name=npz.stem, out_dir=out_dir, **kwargs)


class NicInference(object):
    def __init__(self, model_dir, pack=False, overwrite=False, no_links=False, hmraw=False,
                 compress_batch_size=32, compress_multiproc=False, compress_args=None, num_workers=0,
                 skip_compression=False, flat=None, **kwargs):
        self.model_dir = Path(model_dir)
        self.overwrite = overwrite
        self.pack = pack

        self.no_links = no_links
        self.hmraw = hmraw

        # self.clf_thresholds = None
        # val_results_path = self.model_dir/'eval_validation/results.json'
        # if val_results_path.exists():
        #     results = read_json_dict(self.model_dir/'eval_validation/results.json')
        #     self.clf_thresholds = results.get('validation',{}).get('clf_thresholds',None)
        # else:
        #     print('validation results not found')
        self.args = read_json_dict(self.model_dir/ 'args.json')
        self.args.update(kwargs)
        self.flat = self.args.pop('flat',False)
        if flat is not None:
            self.flat = flat
        self.train_type = self.args['train_type']
        self.target_names = self.args.get('class_names',self.args.get('target_names'))

        net_conf = self.args['net_conf']
        self.code_size = self.args['enc_dim']
        self.fp16 = self.args.get('precision',None)==16
        if compress_args is None:
            self.compress_args = read_json_dict(self.model_dir/ 'compress_args.json')
        elif is_string_or_path(compress_args):
            self.compress_args = read_json_dict(compress_args)
        else:
            self.compress_args = compress_args
        self.compress_args['batch_size'] = compress_batch_size
        # if self.compress_args.get('multiproc',False):
        self.compress_args['multiproc'] = compress_multiproc
        self.skip_compression = skip_compression

        self.model = create_nic_model(**net_conf)
        model_path = find_pl_model_path(self.model_dir)
        load_pl_state_dict(model_path, self.model, replacements={'att_net.attention_a':'att_net.att_m',
                                                                 'att_net.attention_b':'att_net.att_gate',
                                                                 'att_net.attention_c':'att_net.att_last'})

        self.num_workers = num_workers

        # self.device = create_device()

    def _pack(self, slide, mask, out_dir):
        raise ValueError('implement caling packing')

    def _compress(self, slide_path, mask_path, out_dir):
        cargs = self.compress_args.copy()
        cargs.update(dict(out_dir=out_dir, multiproc=False, thumbs=False, overwrite=self.overwrite),
                     data=str(slide_path), mask_dir=str(mask_path))
        out_format = self.compress_args.get('out_format','h5')
        # ok = compress(slide_path, mask_dir=mask_path, **cargs)
        ok = compress(**cargs)
        if not ok:
            raise ValueError('compression of %s failed, params %s' % (str(slide_path), self.compress_args))
        compressed_path = Path(out_dir)/(Path(slide_path).stem+'.'+out_format)
        return compressed_path

    def apply(self, slide_path=None, mask_path=None, compressed_path=None, out_dir=None):
        if compressed_path is None and slide_path is None:
            raise ValueError('either slide or compressed have to be specified')

        if out_dir is None:
            out_dir = Path(self.model_dir)/'apply'
        else:
            out_dir = Path(out_dir)

        if compressed_path is None:
            if mask_path is None:
                print('Warning! no masks')

            if self.pack and mask_path is not None:
                slide_path, mask_path = self._pack(slide_path, mask_path, out_dir/'packed')

            compressed_path = self._compress(slide_path, mask_path, out_dir=out_dir/'compressed')

        slide_name = Path(slide_path).stem
        target_names = self.args['target_names'] if 'target_names' in self.args else self.args['class_names']
        dummy_entry = {DataConf.name_col:slide_name, DataConf.image_col:str(slide_path), DataConf.split_col:'testing'}
        for tn in target_names:
            dummy_entry[tn] = 1
        dummydf = pd.DataFrame([dummy_entry])
        data_conf = DataConf(dummydf, train_type=self.train_type, target_names=target_names)
        compressed_dir = str(Path(compressed_path).parent.absolute())


        ds = CompressedDataset(compressed_dir, data_conf=data_conf, split='testing', flat=False, convert_f32=self.args.get('fp16',None) in [False, None])
        loader = DataLoader(ds, batch_size=1, num_workers=self.num_workers)

        dfp = self._apply_loader(loader, out_dir)

        rec = dfp.to_dict('records')[0]
        rec['compressed'] = str(compressed_path)
        if not self.args.get('no_heatmaps', False):
            hm_writer = HeatmapWriter(npz_dir=out_dir, overwrite=self.overwrite, no_links=self.no_links,
                                      hmraw=self.hmraw, train_type=self.train_type)
            hm_writer.save_hm_and_patches(rec, out_dir, data_conf.get_target_cols())

            hm_path = hm_writer.history_hm_pathes[slide_name][0]
            rec['hm_path'] = str(Path(hm_path).absolute())
        return rec

    def _apply_loader(self, loader, out_dir):
        inf_out_writer = InferenceOutWriter(overwrite=self.overwrite)
        inferencer = Inferencer(self.model, post_fct=TrainType.post_fct(self.train_type), overwrite=self.overwrite,
                                callbacks=[inf_out_writer], fp16=self.fp16)
        dfp = inferencer.apply(loader, out_dir)
        print_df(dfp)
        return dfp

    def apply_config(self, config, overwrite=False, compressed_dir=None, out_dir=None):
        if compressed_dir is None:
            compressed_dir = out_dir/'compressed'
            mkdir(compressed_dir)
        if not self.skip_compression:
            cargs = self.compress_args.copy()
            cargs.update(dict(out_dir=out_dir, thumbs=False, overwrite=self.overwrite))
            # out_format = self.compress_args.get('out_format','h5')
            ok = compress(config, overwrite=overwrite, **cargs)



    def _get_last_model_path(self):
        return find_pl_model_path(self.out_dir, last=True)
        # return Path(self.out_dir)/'last.ckpt'

