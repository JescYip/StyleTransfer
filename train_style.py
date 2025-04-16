import os
from unittest.mock import patch
from tqdm.auto import tqdm
from opt import config_parser
import glob
import torch
import json, random
from renderer import *
from utils import *
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import datetime
from dataLoader import dataset_dict
import sys
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

renderer = OctreeRender_trilinear_fast

class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr += self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr+self.batch]

def InfiniteSampler(n):
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0

class InfiniteSamplerWrapper(torch.utils.data.sampler.Sampler):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31

import torch
import torch.nn.functional as F

def contrastive_loss(z1, z2, temperature=0.07):
    """
    Symmetric InfoNCE Loss (SimCLR-style).
    Args:
        z1: Tensor of shape (B, D) — features of original.
        z2: Tensor of shape (B, D) — features of augmented.
        temperature: Scaling factor.
    Returns:
        Scalar loss.
    """
    device = z1.device
    batch_size = z1.shape[0]

    # Normalize
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    representations = torch.cat([z1, z2], dim=0)  # [2B, D]
    similarity = torch.matmul(representations, representations.T)  # [2B, 2B]
    similarity = similarity / temperature

    # Mask to remove similarity with self
    mask = torch.eye(2 * batch_size, device=device).bool()
    similarity.masked_fill_(mask, float('-inf'))

    # Labels: positives are at fixed offset
    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size),
        torch.arange(0, batch_size)
    ]).to(device)

    loss = F.cross_entropy(similarity, labels)
    return loss

@torch.no_grad()
def render_test(args):
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray

    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.change_to_feature_mod(args.n_lamb_sh, device)
    tensorf.load(ckpt)
    tensorf.eval()
    tensorf.rayMarch_weight_thres = args.rm_weight_mask_thre

    logfolder = os.path.dirname(args.ckpt)

    if args.render_train:
        os.makedirs(f'{logfolder}/{args.expname}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        evaluation_feature(train_dataset, tensorf, args, renderer, args.chunk_size, f'{logfolder}/{args.expname}/imgs_train_all/',
                           N_vis=-1, N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=device)

    if args.render_test:
        os.makedirs(f'{logfolder}/{args.expname}/imgs_test_all', exist_ok=True)
        evaluation_feature(test_dataset, tensorf, args, renderer, args.chunk_size, f'{logfolder}/{args.expname}/imgs_test_all/',
                           N_vis=-1, N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=device)

    if args.render_path:
        c2ws = test_dataset.render_path
        os.makedirs(f'{logfolder}/{args.expname}/imgs_path_all', exist_ok=True)
        evaluation_feature_path(test_dataset, tensorf, c2ws, renderer, args.chunk_size, f'{logfolder}/{args.expname}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=device)

def reconstruction(args):
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    h_rays, w_rays = train_dataset.img_wh[1], train_dataset.img_wh[0]
    ndc_ray = args.ndc_ray

    patch_size = args.patch_size
    logfolder = f'{args.basedir}/{args.expname}'
    os.makedirs(logfolder, exist_ok=True)
    summary_writer = SummaryWriter(logfolder)

    aabb = train_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    nSamples = min(args.nSamples, cal_n_samples(reso_cur, args.step_ratio))

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)
    tensorf.change_to_feature_mod(args.n_lamb_sh, device)
    tensorf.rayMarch_weight_thres = args.rm_weight_mask_thre

    train_dataset.prepare_feature_data(tensorf.encoder)
    feature_files = sorted(glob.glob('/content/features_cached/feature_*.pt'))

    grad_vars = tensorf.get_optparam_groups_feature_mod(args.lr_init, args.lr_basis)
    lr_factor = args.lr_decay_target_ratio ** (1 / args.lr_decay_iters if args.lr_decay_iters > 0 else 1 / args.n_iters)
    optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))
    scaler = GradScaler()

    torch.cuda.empty_cache()
    PSNRs = []
    allrays_stack, allrgbs_stack = train_dataset.all_rays_stack, train_dataset.all_rgbs_stack
    trainingSampler = SimpleSampler(allrays_stack.size(1) * allrays_stack.size(2), args.batch_size)
    frameSampler = iter(InfiniteSamplerWrapper(allrays_stack.size(0)))
    TV_weight_feature = args.TV_weight_feature
    tvreg = TVLoss()

    for iteration in tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout):
        frame_idx = next(frameSampler)
        rays_train = allrays_stack[frame_idx].reshape(-1, 6).to(device)
        features_train = torch.load(train_dataset.feature_paths[frame_idx]).to(device).reshape(-1, 256)

        start_h = np.random.randint(0, h_rays - patch_size + 1)
        start_w = np.random.randint(0, w_rays - patch_size + 1)
        rays_patch = allrays_stack[frame_idx, start_h:start_h+patch_size, start_w:start_w+patch_size, :].reshape(-1, 6).to(device)
        rgbs_patch = allrgbs_stack[frame_idx, start_h:start_h+patch_size, start_w:start_w+patch_size, :].to(device)

        with autocast():
            feature_map, _ = renderer(rays_patch, tensorf, chunk=args.chunk_size, N_samples=nSamples,
                                      white_bg=white_bg, ndc_ray=ndc_ray, render_feature=True, device=device, is_train=True)
            feature_map = feature_map.reshape(patch_size, patch_size, 256)[None].permute(0, 3, 1, 2)
            recon_rgb = tensorf.decoder(feature_map)

            rgbs_patch = rgbs_patch[None].permute(0, 3, 1, 2)
            img_enc = tensorf.encoder(normalize_vgg(rgbs_patch))
            recon_rgb_enc = tensorf.encoder(recon_rgb)

            feature_loss = (F.mse_loss(recon_rgb_enc.relu4_1, img_enc.relu4_1) +
                            F.mse_loss(recon_rgb_enc.relu3_1, img_enc.relu3_1)) / 10
            
            pixel_loss = torch.mean((denormalize_vgg(recon_rgb) - rgbs_patch) ** 2)

            rgb_aug = TF.hflip(rgbs_patch)
            enc_aug = tensorf.encoder(normalize_vgg(rgb_aug))
            z1 = recon_rgb_enc.relu3_1.view(recon_rgb_enc.relu3_1.shape[0], -1)
            z2 = enc_aug.relu3_1.reshape(enc_aug.relu3_1.shape[0], -1)
            cl_loss = contrastive_loss(z1, z2)

            total_loss = pixel_loss + feature_loss + 0.5 * cl_loss
            if TV_weight_feature > 0:
                TV_weight_feature *= lr_factor
                total_loss += tensorf.TV_loss_feature(tvreg) * TV_weight_feature

        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        summary_writer.add_scalar('train/mse_pixel', pixel_loss.item(), iteration)
        summary_writer.add_scalar('train/mse_feature', feature_loss.item(), iteration)
        summary_writer.add_scalar('train/contrastive_loss', cl_loss.item(), iteration)

    tensorf.save(f'{logfolder}/{args.expname}.th')

if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)
    torch.backends.cudnn.benchmark = True
    args = config_parser()
    print(args)
    if args.render_only:
        render_test(args)
    else:
        reconstruction(args)
