from pathlib import Path

import SimpleITK as sitk

from wsilearn.wsi.wsd_image import ArrayImageWriter


def load_mha(path, meta=False):
    image = sitk.ReadImage(str(path), imageIO="MetaImageIO")
    arr = sitk.GetArrayFromImage(image)
    if meta:
        keys = list(image.GetMetaDataKeys())
        mdata = {k:image.GetMetaData(k) for k in keys}
        return arr, mdata
    return arr

def save_mha(arr, path):
    image = sitk.GetImageFromArray(arr)
    sitk.WriteImage(image, str(path))

def save_mha_as_tif(mha_path, out_dir, spacing=0.5):
    print('saving %s as tif' % mha_path)
    arr, keys = load_mha(mha_path, meta=True)
    print('mha shape:', arr.shape)
    print(keys)

    out_path = Path(out_dir)/(Path(mha_path).stem+'.tif')
    writer = ArrayImageWriter()
    writer.write_array(arr, out_path, spacing=spacing)
    # write_array(arr, out_path, out_spacing=spacing)

if __name__ == '__main__':
    # path=...
    # arr = load_mha(path)
    # print(arr.shape)
    # save_mha_as_tif(path, out_dir='./out/')
    pass