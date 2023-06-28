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

Compression script: 
wsilearn.compress.compress.py

Classification script:
wsilearn.train_nic_pl.py


More documentation to follow soon


[1] Mormont, Romain, Pierre Geurts, and Raphaël Marée. "Multi-task pre-training of deep neural networks for digital pathology." IEEE journal of biomedical and health informatics 25.2 (2020): 412-421

[2] Ciga, Ozan, Tony Xu, and Anne Louise Martel. "Self supervised contrastive learning for digital histopathology." Machine Learning with Applications 7 (2022): 100198.