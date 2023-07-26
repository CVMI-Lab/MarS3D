<div align="center">

<h1>MarS3D: A Plug-and-Play Motion-Aware Model for Semantic Segmentation on Multi-Scan 3D Point Clouds</h1>

<div>
    <a href="https://github.com/jliu-ac" target="_blank">Jiahui Liu</a><sup>1*</sup>,</span>
    <a href="https://github.com/jUstin-crchang" target="_blank">Chirui Chang</a><sup>1*</sup>,</span>
    <a href="https://scholar.google.com/citations?user=n1JW-jYAAAAJ&hl=en" target="_blank">Jianhui Liu</a><sup>1</sup>,</span>
    <a href="https://xywu.me/" target="_blank">Xiaoyang Wu</a><sup>1</sup>,</span>
    <a>Lan Ma</a><sup>2</sup>,</span>
    <a href="https://xjqi.github.io/" target="_blank">Xiaojuan Qi</a><sup>1&#8224</sup>,</span>  
</div>

<div>
    <sup>1</sup>The University of Hong Kong&emsp;
    <sup>2</sup>TCL AI Lab
</div>

<div>
    *equal contribution&emsp;
    <sup>+</sup>corresponding author
</div>

**CVPR 2023**

<img src="assets/demo.gif" width="75%"/>

MarS3D is a plug-and-play motion-aware module for semantic segmentation on multi-scan 3D point clouds. Extensive experiments show
that MarS3D can improve the performance of the baseline model by a large margin.

[video](https://www.youtube.com/watch?v=PPPyZkwvsvs) | [arXiv](https://arxiv.org/abs/2307.09316) | [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_MarS3D_A_Plug-and-Play_Motion-Aware_Model_for_Semantic_Segmentation_on_Multi-Scan_CVPR_2023_paper.pdf)

</div>

## Interesting Features

- A **plug-and-play** module, which can be flexibly integrated with mainstream single-scan segmentation models.
- A **Cross-Frame Feature Embedding** for temporal information preservation.
- A **BEV-based Motion-Aware Feature** Learning module to exploit temporal information and enhance the model’s motion awareness.

## Performance

- Performance improvement over baselines on SemanticKITTI public multi-scan validation set.
  
    | Method | mIoU | #param |
    | :---: |:---: |:---: |
    |SPVCNN   | 49.70%| 21.8M|
    |SPVCNN+MarS3D  | **54.66%** | 21.9M|
    |SparseUNet  | 48.99%| 39.2M|
    |SparseUNet+MarS3D  | **54.64%** | 39.3M|
    |MinkUNet  | 48.47%| 37.9M|
    |MinkUNet+MarS3D  | **54.71%** | 38.0M|

- Comparison with the state-of-the-art models on SemanticKITTI multi-scan benchmark.
  
    | Method | mIoU |
    | :---: |:---: |
    |SpSequenceNet   | 43.1%|
    |TemporalLidarSeg   | 47.0%|
    |TemporalLatticeNet   | 47.1%|
    |Meta-RangeSeg    | 49.5%|
    |KPConv  | 51.2%|
    |SPVCNN   | 49.2%|
    |SPVCNN+MarS3D   | 52.7%|

## Getting Started

### Installatioon

#### System requirements

- Ubuntu: 18.04+
- CUDA: 11.x

#### Python dependencies

```bash
conda create -n mars3d python=3.9 -y
conda activate mars3d
conda install ninja -y
conda install pytorch torchvision torchaudio cudatoolkit -c pytorch -y  # follow the instructions from the official pytorch website
conda install -c anaconda h5py pyyaml -y
conda install -c conda-forge sharedarray tensorboardx yapf addict einops scipy plyfile -y
pip install tqdm
pip install numpy==1.23 # for fixing version conflict
```

#### Backbone dependencies

```bash
# Please follow the official instructions from the official repo for different backbones
# Minkowski Engine: https://github.com/NVIDIA/MinkowskiEngine

# spconv: https://github.com/traveller59/spconv

# torchsparse: https://github.com/mit-han-lab/torchsparse

# torchsparse: an optional installation without sudo apt install
conda install google-sparsehash -c bioconda
export C_INCLUDE_PATH=${CONDA_PREFIX}/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=${CONDA_PREFIX}/include:CPLUS_INCLUDE_PATH
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git
```

#### Dataset Preparation
**Semantic-Kitti**
- Download the [Semantic-Kitti](http://www.semantic-kitti.org/dataset.html#download)
- The dataset file structure is listed as follows:
```
./
├── 
├── ...
└── path_to_semantic_kitti/
    ├──sequences
        ├── 00/           
        │   ├── velodyne/	
        |   |	├── 000000.bin
        |   |	├── 000001.bin
        |   |	└── ...
        │   └── labels/ 
        |       ├── 000000.label
        |       ├── 000001.label
        |       └── ...
        ├── 08/ # 08 is the validation set
        ├── 11/ # 11-21 is the test set
        └── ...
        └── 21/
```
- Symlink the paths to them as follows:
```
mkdir data
ln -s /path_to_semantic_kitti data/semantic_kitti
```
## Run
#### Training
```
sh scripts/train.sh -p INTERPRETER_PATH -d DATASET -c CONFIG_NAME -n EXP_NAME
```
for example:
```
sh scripts/train.sh -p python -d semantic_kitti -c spvcnn-mh -n mars3d-spvcnn
```
#### Inference
*For calling test script, exp folder generated in training process start by training script is needed.*

*Furthermore, currently, the weight path should be specified in test.py*
```
sh scripts/test.sh -p INTERPRETER_PATH -d DATASET -c CONFIG_NAME -n EXP_NAME
```
for example:
```
sh scripts/test.sh -p python -d semantic_kitti -c spvcnn-mh -n mars3d-spvcnn
```

## Citation
If you find this project useful in your research, please consider cite:
```bibtex
@inproceedings{liu2023mars3d,
  title={MarS3D: A Plug-and-Play Motion-Aware Model for Semantic Segmentation on Multi-Scan 3D Point Clouds},
  author={Liu, Jiahui and Chang, Chirui and Liu, Jianhui and Wu, Xiaoyang and Ma, Lan and Qi, Xiaojuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9372--9381},
  year={2023}
}
```
