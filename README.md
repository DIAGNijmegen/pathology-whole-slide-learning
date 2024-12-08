
# pathology-whole-slide-learning
Framework for weak whole slide learning with attention: compress and classify whole slide images.  
Requires ASAP and the wholeslidedata package and optionally pathology-whole-slide-packer.

Preprocessing: For slides bigger then ~50k pixels in height/width packing is recommended,  
as otherwise the gpu out of memory might occur.

Steps:
1. Compress the slides
2. Train Classifier
3. Evaluate/apply classifier

For compression several pretrained encoders are supported, i.e.
- res50: an ImageNet pretrained ResNet50
- mtdp_res50: Histologically pretrained encoder by Mormont et al. [1] Requires the multitask-dipath  
  library to be present in the PYTHONPATH (https://github.com/waliens/multitask-dipath)
- histossl: Histologically pretrained ResNet18 encoder by Ciga et al. [2]. Requires to  
  download their model to ~/.torch/models

More documentation to follow soon

## Installation

Requires python 3.10 and pip<=24.0

pip install -r requirements.txt  
The training code has a dependency to pytorch-lightning 1.7.7.

## Usage

### Tissue Masks
The framework expects tissue masks as pyramidal images. They can be create, for example, via the script `create_tissue_masks`from the library https://github.com/DIAGNijmegen/pathology-whole-slide-packer.

### Compression
First, the slides are compressed with an encoder at specific spacing. Here, a bash template example:

    data=<slide_directory or csv>
    mask_dir=<tissue masks, e.g. <slide_name>_tissue.tif>
    out_dir=${encoder}_sp${spacing_str}
    encoder=res50
    spacing=0.5; spacing_str="sp05"
    patch_size=224; stride=224
    batch_size=512
    
    python3 wsilearn/compress/compress.py \
    --data=$data --mask_dir=$masks --out_dir=$out_dir \
    --encoder=${encoder} --mask_spacing=4 \
    --batch_size=${batch_size} --spacing=${spacing} --stride=${stride} --patch_size=${patch_size} \
    --clear_locks

If the script is run on the same slides in parallel, the flag clear_locks should be removed. In the end, a csv
with a summary of the compressed slides is created (necessary for the next step).

### Training
#### Data configuration file
The data needs to be split in at least a training and validation partition via csv. The csv should have the columns: name, split (i.e. training, validation) and one column for each target class in one-hot encoding.
A minimal configuration could look like this:

*name,split,Normal,Cancer*  
file1,training,0,1  
file2,training,1,0  
file3,validation,0,1  
file4,validation,1,0  

There can be more splits in the configuration, which are then treated as different testing subsets and evaluated after training end. There can be also a column *category*  can be used to compute additional evaluation metrics for subsets of a split.

The configuration may contain other columns. This allows to specify different targets while keeping the same data configuration. The (hot-encoded) target class columns are given via the class_names parameter separated by commas. 
#### Example:

    data_config=<data_config.csv>; preprocess_dir=<compressed out_dir>
    encoder=res50; enc_dim=1024
    dropout=0.25
    seed=1
    class_names=Normal,Cancer
    monitor=val_auc #also supported val_loss and other metrics shown in the generated history.csv
    python3 wsilearn/train_nic.py	\
    	 --data_config=$data --preprocess_dir=$preprocess_dir --out_dir=$out_dir \
    	 --precision=16 --autoname --enc_dim=$enc_dim \
    	 --train_type=clf --class_names=$class_names \
    	 --monitor=$monitor \
    	 --net_conf.name=attnet --net_conf.dropout=$dropout \
    	 --num_workers=4 --seed=$seed \
    	 --no_heatmaps --no_out

Important additional parameters:

- pathes_csv: (name,path) with the pathes to the slides so that heatmaps can be created if the script is run without the no_heatmaps flat can be created
- net_conf.topk: number of patches with highest attention to compute the predictions
- balance: sample compressed slides inverse to their frequency
- early_patience: number of epochs for early stopping (default is 25)

#### Multi-label tasks
Multi-label classification is supported with the same configuration format via: --train_type=multilabel

### Evaluation
The trained models can be evaluated on a new dataset via the evaluation script:

Example:

    python3 wsilearn/eval_nic.py \
    --model_dir=${model_dir} --data_config=${data_config} --preprocess_dir=$preprocess_dir \
    --out_dir=$out_dir --pathes_csv=$pathes_csv --num_workers=4

## Encoders
[1] Mormont, Romain, Pierre Geurts, and Raphaël Marée. "Multi-task pre-training of deep neural networks for digital pathology." IEEE journal of biomedical and health informatics 25.2 (2020): 412-421

[2] Ciga, Ozan, Tony Xu, and Anne Louise Martel. "Self supervised contrastive learning for digital histopathology." Machine Learning with Applications 7 (2022): 100198.