## Installation
> Tested on Ubuntu 20.04 + Pytorch 1.12.1

Install environment:
```
conda create -n StyleRF python=3.9
conda activate StyleRF
pip install torch torchvision
pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg kornia lpips tensorboard
```

## Datasets
Please put the datasets in `./data`. You can put the datasets elsewhere if you modify the corresponding paths in the configs.

### 3D scene datasets
* [nerf_synthetic](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) 
* [llff](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)
### Style image dataset
* [WikiArt](https://www.kaggle.com/datasets/ipythonx/wikiart-gangogh-creating-art-gan)

To generate stylized novel views:
```
bash scripts/test_style.sh [GPU ID]
```
The rendered stylized images can then be found in the directory under the checkpoint's path.

## Training
> Current settings in `configs` are tested on one NVIDIA RTX A5000 Graphics Card with 24G memory. To reduce memory consumption, you can set `batch_size`, `chunk_size` or `patch_size` to a smaller number.

We follow the following 3 steps of training:
### 1. Train original TensoRF
This step is for reconstructing the density field, which contains more precise geometry details compared to mesh-based methods. You can skip this step by directly downloading pre-trained checkpoints provided by [TensoRF checkpoints](https://1drv.ms/u/s!Ard0t_p4QWIMgQ2qSEAs7MUk8hVw?e=dc6hBm).

The configs are stored in `configs/llff.txt` and `configs/nerf_synthetic.txt`. For the details of the settings, please also refer to [TensoRF](https://github.com/apchenstu/TensoRF). The checkpoints are stored in `./log` by default.

You can train the original TensoRF by:
```
bash script/train.sh [GPU ID]
```

### 2. Feature grid training stage
This step is for reconstructing the 3D gird containing the VGG features.

The configs are stored in `configs/llff_feature.txt` and `configs/nerf_synthetic_feature.txt`, in which `ckpt` specifies the checkpoints trained in the **first** step. The checkpoints are stored in `./log_feature` by default.

Then run:
```
bash script/train_feature.sh [GPU ID]
```


### 3. Stylization training stage 
This step is for training the style transfer modules.

The configs are stored in `configs/llff_style.txt` and `configs/nerf_synthetic_style.txt`, in which `ckpt` specifies the checkpoints trained in the **second** step. The checkpoints are stored in `./log_style` by default.

Then run:
```
bash script/train_style.sh [GPU ID]
```

---
## Training on 360 Unbounded Scenes
The code for training StyleRF on the Tanks&Temples dataset is available on the `360` branch. To access it, run `git checkout 360`.

## Acknowledgments
Our implementation builds upon the official [StyleRF](https://github.com/Kunhao-Liu/StyleRF) codebase. We thank the authors for releasing such a well-structured and effective framework.
