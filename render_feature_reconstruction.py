import os, torch
from dataLoader import dataset_dict
from models.tensoRF import TensorVMSplit
from renderer import denormalize_vgg
import imageio

@torch.no_grad()
def render_from_feature_volume(ckpt_path, datadir, feature_dir, save_dir, device='cuda'):
    os.makedirs(save_dir, exist_ok=True)
    ckpt = torch.load(ckpt_path, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = TensorVMSplit(**kwargs)
    tensorf.load(ckpt)
    tensorf.eval()

    dataset = dataset_dict['own_data']
    test_dataset = dataset(datadir, split='test', downsample=1.0, is_stack=True)
    W, H = test_dataset.img_wh

    for i, path in enumerate(sorted(os.listdir(feature_dir))):
        if not path.endswith(".pt"): continue
        feature = torch.load(os.path.join(feature_dir, path)).permute(2, 0, 1).unsqueeze(0).to(device)
        rgb = denormalize_vgg(tensorf.decoder(feature)).permute(0, 2, 3, 1).squeeze(0).clamp(0, 1).cpu().numpy()
        rgb = (rgb * 255).astype('uint8')
        imageio.imwrite(os.path.join(save_dir, f'recon_{i:03d}.png'), rgb)
    print(f"Done: the reconstructed image has been saved to {save_dir}")

render_from_feature_volume(
    ckpt_path='/content/StyleTransfer/log_feature/own_feature/own_feature.th',
    datadir='/content/drive/MyDrive/data/nerf_data',
    feature_dir='/content/features_cached',
    save_dir='/content/reconstructed_rgb'
)
