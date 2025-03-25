expname=lego_feature
CUDA_VISIBLE_DEVICES=$1 python train_feature.py \
--config configs/nerf_synthetic_feature.txt \
--datadir ./data/nerf_synthetic/lego \
--expname $expname \
--ckpt ./log/lego_base/latest.tar
