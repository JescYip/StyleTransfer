import torch,cv2
from torch.utils.data import Dataset
import json
from tqdm import tqdm
import os
import glob
from PIL import Image
from torchvision import transforms as T


from .ray_utils import *


class BlenderDataset(Dataset):
    def __init__(self, datadir, split='train', downsample=1.0, is_stack=False, N_vis=-1):

        self.N_vis = N_vis
        self.root_dir = datadir
        self.split = split
        self.is_stack = is_stack
        self.img_wh = (int(800/downsample),int(800/downsample))
        self.define_transforms()

        self.scene_bbox = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.read_meta()
        self.define_proj_mat()

        self.white_bg = True
        self.near_far = [2.0,6.0]
        
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        self.downsample=downsample

    def read_depth(self, filename):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
        return depth
    
    def read_meta(self):

        with open(os.path.join(self.root_dir, f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)

        w, h = self.img_wh
        self.focal = 0.5 * 800 / np.tan(0.5 * self.meta['camera_angle_x'])  # original focal length
        self.focal *= self.img_wh[0] / 800  # modify focal length to match size self.img_wh


        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, [self.focal,self.focal])  # (h, w, 3)
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
        self.intrinsics = torch.tensor([[self.focal,0,w/2],[0,self.focal,h/2],[0,0,1]]).float()

        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_masks = []
        self.all_depth = []
        self.downsample=1.0

        img_eval_interval = 1 if self.N_vis < 0 else len(self.meta['frames']) // self.N_vis
        idxs = list(range(0, len(self.meta['frames']), img_eval_interval))
        for i in tqdm(idxs, desc=f'Loading data {self.split} ({len(idxs)})'):#img_list:#

            frame = self.meta['frames'][i]
            pose = np.array(frame['transform_matrix']) @ self.blender2opencv
            c2w = torch.FloatTensor(pose)
            self.poses += [c2w]

            image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
            self.image_paths += [image_path]
            img = Image.open(image_path)
            
            if self.downsample!=1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (4, h, w)
            img = img.view(4, -1).permute(1, 0)  # (h*w, 4) RGBA
            self.all_masks.append(img[:, -1:].reshape(h,w,1)) # (h, w, 1) A
            img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
            self.all_rgbs += [img]


            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)

        
        self.all_masks = torch.stack(self.all_masks) # (n_frames, h, w, 1)
        self.poses = torch.stack(self.poses)
        all_rays = self.all_rays
        all_rgbs = self.all_rgbs

        self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w,6)
        self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w,3)

        if self.is_stack:
            self.all_rays_stack = torch.stack(all_rays, 0).reshape(-1,*self.img_wh[::-1], 6)  # (len(self.meta['frames]),h,w,6)
            avg_pool = torch.nn.AvgPool2d(4, ceil_mode=True)
            self.ds_all_rays_stack = avg_pool(self.all_rays_stack.permute(0,3,1,2)).permute(0,2,3,1) # (len(self.meta['frames]),h/4,w/4,6)
            self.all_rgbs_stack = torch.stack(all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)
    
    @torch.no_grad()
    def prepare_feature_data(self, encoder, save_dir='/content/features_cached'):
        os.makedirs(save_dir, exist_ok=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = encoder.to(device)
        for name, module in encoder.named_children():
            module.to(device)

        print('====> prepare_feature_data (low memory + save per frame)...')
        frames_num, h, w, _ = self.all_rgbs_stack.shape

        for i in tqdm(range(frames_num)):
            save_path = os.path.join(save_dir, f'feature_{i:03d}.pt')
            if os.path.exists(save_path):
                continue  

            rgb = self.all_rgbs_stack[i].to(device)
            rgb = rgb.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
            rgb = normalize_vgg(rgb)

            feature = encoder(rgb).relu3_1
            feature = T.functional.resize(feature, size=(h, w), interpolation=T.InterpolationMode.BILINEAR)
            feature = feature.squeeze(0).permute(1, 2, 0).cpu()  # [H, W, C]

            torch.save(feature, save_path) 
            del rgb, feature
            torch.cuda.empty_cache()
        self.feature_paths = sorted(glob.glob('/content/features_cached/feature_*.pt'))
        print('====> feature extraction and save DONE.')


    def define_transforms(self):
        self.transform = T.ToTensor()
        
    def define_proj_mat(self):
        self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses)[:,:3]

    def world2ndc(self,points,lindisp=None):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)
        
    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):

        if self.split == 'train':  # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}

        else:  # create data for each image separately

            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]
            mask = self.all_masks[idx] # for quantity evaluation

            sample = {'rays': rays,
                      'rgbs': img,
                      'mask': mask}
        return sample