# DF4LCZ

This repository hosts the code for the DF4LCZ model which is a sam-empowered data fusion framework for scene-level local climate zone classification. The relevant paper is [DF4LCZ: A SAM-Empowered Data Fusion Framework for Scene-Level Local Climate Zone Classification](https://ieeexplore.ieee.org/abstract/document/10556641?casa_token=fPD1keUeTw4AAAAA:z8iuo_WSxWuCd2qQI9jmF-q8L_R9EorHqqMHUZAAADKLSByRw1hej2wcW07g0Rgnsjvl75nuE4DYH7I)

## Usage

### Install
- Python 3.8
- Tensorflow 2.11.0 
- Keras 2.11.0 
- Spektral 1.3.0
- albumentations 1.3.1

### Train and test

- Change the current directory to `DF4LCZ`.
- For the Sentinel-2 stream, run `single_fix_augment.py` using

```Bash
python ./single_fix_augment.py --model 'resnet11_3D' --batch_size 64 --initial_lr 0.002 --decay_factor 0.4 --patience 40 --epoch 100
```
- For the Google Earth stream, run `GNN_train.py` using

```Bash
python ./GNN_train.py
```
- For the DF4LCZ classification, 
    - Add the weights file paths derived from the two streams into the `fusion.py`,
    - and run `fusion.py`

## LCZC-GES2 DATASET

- The LCZC-GES2 dataset comprises 19,088 pairs of image patches. Each pair includes a Google Earth RGB image and a Sentinel-2 multispectral image patch. The data can be found in the folders `sen2_img_patches` and `gg_nodes_refine`.(`gg_nodes_refine` contains all graphes after the graph construction procedure, the original google patches please refer to [Google drive files](https://drive.google.com/file/d/1DpjqUAXr-At3qAoka6Xfdv7w1s0Jkn3c/view?usp=drive_link))

- The folder `patches_split` contains the spliting results from two sampling strategies. The file `partition_polygons_1125.npz` represents one strategy, named “splitting the polygon pool,” while `partition_random.npz` represents the other strategy, named “splitting the sample pool.”

## Latest Updates 

- [2024/05/07]: Uploaded a batch of code, including the implementation of DF4LCZ and several comparative models. 

## Citation

If you use this code in your research, please consider citing the following paper:

```
@ARTICLE{10556641,
  author={Wu, Qianqian and Ma, Xianping and Sui, Jialu and Pun, Man-On},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={DF4LCZ: A SAM-Empowered Data Fusion Framework for Scene-Level Local Climate Zone Classification}, 
  year={2024},
  volume={62},
  number={},
  pages={1-16},
  keywords={Feature extraction;Internet;Earth;Task analysis;Image segmentation;Spatial resolution;Meteorology;Data fusion;local climate zone (LCZ) classification;segment anything model (SAM)},
  doi={10.1109/TGRS.2024.3414143}}
```

## Contact

For any inquiries or further information, please contact me.

## Other Works
There are some other works in our group:

[SAM-Assisted Remote Sensing Imagery Semantic Segmentation with Object and Boundary Constraints](https://arxiv.org/abs/2312.02464)(TGRS under review)[Code](https://github.com/sstary/SSRS)