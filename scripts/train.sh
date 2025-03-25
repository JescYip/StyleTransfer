expname=lego_base

CUDA_VISIBLE_DEVICES=$1 python train.py \
--config configs/nerf_synthetic.txt \
--datadir ./data/nerf_synthetic/lego \
--expname $expname
