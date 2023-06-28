import h5py
import numpy as np
from scipy.ndimage import distance_transform_edt

from wsilearn.utils.cool_utils import *
from wsilearn.utils.df_utils import print_df, df_merge_check
from wsilearn.utils.files_utils import FilesInfo
from wsilearn.utils.path_utils import PathUtils
from wsilearn.utils.simpleitk_utils import save_mha
import pandas as pd

from wsilearn.utils.cool_utils import print_mp, print_mp_err
from wsilearn.wsi.wsd_image import write_array

print = print_mp

def plot_feature_map(features, idx=None, name=None, chw=True, save_dir=None, show=False, pad=1):
    """
    Preview of the featurized WSI. Draws a grid where each small image is a feature map. Normalizes the set of feature
    maps using the 3rd and 97th percentiles of the entire feature volume. Includes these values in the filename.

    Args:
        features: numpy array with format [c, x, y].
        output_path (str): path pattern of the form '/path/tumor_001_90_none_{f_min:.3f}_{f_max:.3f}_features.png'

    """

    # without copy() it modifies features!!
    features = np.copy(features)
    if not chw and len(features.shape)>2:
        features = features.transpose(2,0,1)

    if idx is not None:
        features = features[idx]

    if len(features.shape)==2:
        features = features[np.newaxis,:,:]

    # Get range for normalization
    f_min = np.percentile(features[features != 0], 3)
    f_max = np.percentile(features[features != 0], 97)

    # Detect background (estimate)
    features[features == 0] = np.nan

    # Normalize and clip values
    features = (features - f_min) / (f_max - f_min + 1e-6)
    features = np.clip(features, 0, 1)

    # Add background
    features[features == np.nan] = 0.5

    # Make batch
    data = features[:, np.newaxis, :, :].transpose(0, 2, 3, 1)

    # Make grid
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0), (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=0.0)
    padding = ((0, 0), (pad, pad), (pad, pad)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=0.5)

    # Tile the individual thumbnails into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    # Map the normalized data to colors RGBA
    cmap = plt.cm.jet
    norm = plt.Normalize(vmin=0, vmax=1)
    image = cmap(norm(data[:, :, 0]))
    plt.axis('off')

    #join(output_dir, filename.format(item='{rot_deg}_{flip}_{f_min}_{f_max}_features.png')
    if save_dir is not None and name is not None:
        ensure_dir_exists(save_dir)
        out_path = Path(save_dir)/(name+f'__{f_min:.3f}_{f_max:.3f}.jpg')
        plt.imsave(str(out_path), image)

    if show:
        plt.tight_layout()
        plt.imshow(image)
        plt.show()


class CompressedInfo(FilesInfo):

    width_col = 'width'
    height_col = 'height'
    code_size_col = 'code_size'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # if self.df is None or len(self.df)==0:
        #     raise ValueError('didnt find compresse infos!', args, kwargs)
        args_str = str(args) if len(args)>0 else '' +str(kwargs) if len(kwargs)>0 else ''
        if self.df is not None:
            print('CompressedInfo with %d entries from %s' % (len(self.df), args_str))
        else:
            print('no compressed infos in %s' % args_str)


    def _get_infos_path(self, cdir):
        return Path(cdir)/'compressed_infos.csv'

    def get_code_size(self, default=None):
        try:
            code_sizes = self.df[CompressedInfo.code_size_col].values
        except:
            print('warning: apparently no code_size_col %s in df:' % (self.code_size_col))
            print_df(self.df.head())
            if default is None:
                raise
            else:
                print('returning default value %d' % default)
                return default
        if len(list(set(code_sizes)))!=1:
            raise ValueError('wrong code sizes', code_sizes)
        return int(code_sizes[0])

    def verify(self, **kwargs):
        names = PathUtils.list_pathes(self.compressed_dir, ret='stem', **kwargs)
        dfn = pd.DataFrame(dict(name=names))
        dfm, dfl_surp, dfr_surp = df_merge_check(self.df, dfn, left='name', right='name', left_title='config', right_title='compressed-csv')
        assert len(dfl_surp)==0
        assert len(dfr_surp)==0

class CompressedSlide(object):
    def __init__(self, path, hwc, cache_dir=None, overwrite=False):
        self.path = path
        self.cache_dir = cache_dir #todo: use
        self.format = self.__class__.__name__
        self.hwc = hwc
        self.overwrite = overwrite

    def read(self, flat):
        """ returns the compresed features, either fat or in compressed slide and other - dictionary with
         other infos including a distance_map (if not flat), """
        raise ValueError('implement read!')

    def show(self, data=None, idx=None, **kwargs):
        # data, infos = H5Features.read_features(self.path)
        if data is None:
            data, infos = self.read(flat=False)
        print(data.shape)
        plot_feature_map(data, idx=idx, chw=not self.hwc, name=Path(self.path).stem, **kwargs)



class CompressedNpz(CompressedSlide):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, hwc=True)
        raise ValueError('check whether npz-format is hwc or not')

    def write(self, result, **kwargs):
        # todo: working_dir
        out_path = str(self.path).replace('.npy', '.npz')

        # plot_feature_map(result, out_path, chw=False)
        # dm_path = Path(out_path).parent /(Path(out_path).stem+ '_distance_map.npy')
        distance_map = compute_single_distance_map(result, hwc=True)
        # np.save(str(dm_path), distance_map)
        return np.savez(str(out_path), features=result, shape=result.shape, distance_map=distance_map, **kwargs)

    def read(self, flat=False):
        if flat: raise ValueError('implement')
        npfile = np.load(str(self.path))
        features = npfile['features']
        other = {k:npfile[k] for k in npfile.keys() if k!='features'}
        return features, other


def save_hdf5(out_dir, mode='w', overwrite=False, **kwargs):
    if Path(out_dir).exists() and not overwrite:
        raise ValueError('%s exists already!' % out_dir)

    file = h5py.File(out_dir, mode)
    #saving non-interables as meta data
    metadata = {k:v for k, v in kwargs.items() if not is_iterable(v)}
    m = file.create_dataset('metadata', data=json.dumps(metadata, cls=JsonNpEncoder))
    #metadata = json.loads(file['metadata'][()])

    arr_map = {k:v for k, v in kwargs.items() if is_iterable(v)}
    for key, val in arr_map.items():
        data_shape = val.shape
        data_type = val.dtype
        chunk_shape = (1, ) + data_shape[1:]
        maxshape = (None, ) + data_shape[1:]
        dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
        dset[:] = val

    file.close()
    return out_dir

class CompressedH5C(CompressedSlide):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, hwc=True)

    def write(self, result, stride, **kwargs):
        fshape = np.array(result.shape)
        features, coords = H5Features.create_features_coords(result, stride)
        # plot_feature_map(result, out_path, chw=False)
        # dm_path = Path(out_path).parent/(Path(out_path).stem+ '_distance_map.npy')
        distance_map = compute_single_distance_map(result, hwc=True)
        save_dict = dict(features=features, distance_map=distance_map, coords=coords, stride=stride,
                         width=fshape[1], height=fshape[0], code_size=fshape[2], shape=fshape,
                         w_dim=0, h_dim=1, code_dim=2, n=len(features), **kwargs)
        return save_hdf5(str(self.path), overwrite=self.overwrite, **save_dict)

    def read(self, flat=True):
        return H5Features.read_features(self.path, stitch=flat==False)

    def info(self):
        try:
            info = H5Features.get_info(self.path)
            info[CompressedInfo.name_col] = Path(self.path).name
            return info
        except:
            print('failed getting info for %s' % str(self.path))
            raise

class CompressedTif(CompressedSlide):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, hwc=True)

    def write(self, result, stride, spacing, **kwargs):
        result = result.squeeze()
        # result = result[:-1,:-1]
        result = result[1:, 1:]  # otherwise doesnt work in asap for some reason...
        out_spacing = spacing * self.stride

        return write_array(result, self.path, out_spacing)

class CompressedMha(CompressedSlide):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, hwc=True)
        raise ValueError('check whether mha-format is hwc or not')

    def write(self, result, **kwargs):
        return save_mha(result, self.path)

def compressed_slide_from_ending(path_ending, **kwargs) -> CompressedSlide:
    ending = str(path_ending).split('.')[-1]
    if ending == 'tif':
        return CompressedTif(**kwargs)
    elif ending in ['npz','npz']:
        return CompressedNpz(**kwargs)
    elif ending in ['mha','mhz']:
        return CompressedMha(**kwargs)
    elif ending in ['h5']:
        return CompressedH5C(**kwargs)
    else:
        raise ValueError('unknown features format %s' % path_ending)


def compute_single_distance_map(features, hwc=False, perc=98):
    features = np.copy(features)

    axis = -1
    if not hwc: #chw
        axis = 0

    # Binarize
    features = features.std(axis=axis)
    features[features != 0] = 1

    # Distance transform
    distance_map = distance_transform_edt(features)
    distance_map = distance_map / np.max(distance_map)
    distance_map = np.square(distance_map)

    distance_map[distance_map < np.percentile(distance_map, perc)] = 0

    distance_map = distance_map / np.sum(distance_map)

    return distance_map.astype(np.float32)


class H5Features(object):
    """ Saving features as h5 """
    @staticmethod
    def get_fshape_from_coords(coords, flat, downscale=256):
        xmax = coords[:, 0].max() // downscale + 1
        ymax = coords[:, 1].max() // downscale + 1
        fshape = (ymax, xmax, flat.shape[-1])
        return fshape

    @staticmethod
    def get_fshape(path):
        file = h5py.File(str(path), "r")
        if 'shape' in file:
            features_shape = file['shape'][()]
            # features_shape = file['shape'].value
        else: #original clam
            # coords = file['coords'].value
            coords = file['coords'][()]
            features_shape = H5Features.get_fshape_from_coords(coords, file['features'])
        file.close()
        return features_shape

    @staticmethod
    def get_info(path):
        if isinstance(path, (str, Path)):
            file = h5py.File(str(path), "r")
            close = True
        else:
            file = path
            close = False

        if 'shape' in file:
            infos = json.loads(file['metadata'][()])
            infos['shape'] = file['shape'][()]
        else:
            #file['coords'].value
            coords = file['coords'][()]
            features_shape = H5Features.get_fshape_from_coords(coords, file['features'])
            infos = dict(shape=features_shape)
        if close:
            file.close()
        return infos

    @staticmethod
    def stitch_features_from(flat, coords, downscale=256, fshape=None):
        if fshape is None:
            fshape = H5Features.get_fshape_from_coords(coords, flat, downscale=downscale)
        dtype = flat.dtype
        stitched = np.zeros((fshape), dtype=dtype)
        # print('STITCH FSHAPE:', fshape, 'coord-max:',coords.max(axis=0))
        for i,coord in enumerate(coords):
            stitched[coord[1]//downscale, coord[0]//downscale] = flat[i]
        return stitched

    @staticmethod
    def read_features(path, stitch=True):
        file = h5py.File(str(path), "r")
        features = file['features'][()]
        infos = H5Features.get_info(file)
        coords = file['coords'][()]
        infos['coords'] = coords

        file.close()

        if stitch:
            features = H5Features.stitch_features_from(features, coords, fshape=infos.get('shape',None),
                                                       downscale=infos['stride'])

        return features, infos

    @staticmethod
    def create_features_coords(result, downscale):
        """  [col,row] at original processing spacing """
        finds = result.std(axis=-1) != 0
        features = result[finds, :]
        coords = np.array(np.where(finds)).T*downscale
        coords_ = np.zeros_like(coords)
        coords_[:,0] = coords[:,1]
        coords_[:,1] = coords[:,0]
        coords = coords_
        return features, coords

