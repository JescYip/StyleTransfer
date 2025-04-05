# render_style.py
import os, torch, argparse
from dataLoader import dataset_dict
from models.tensoRF import TensorVMSplit
from renderer import evaluation_feature, OctreeRender_trilinear_fast
from utils import denormalize_vgg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/nerf_synthetic_style.txt')
    parser.add_argument('--datadir', type=str, required=True, help='Path to dataset')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to .th model file')
    parser.add_argument('--expname', type=str, required=True, help='Name for output folder')
    parser.add_argument('--N_vis', type=int, default=100, help='Number of test views to render')
    parser.add_argument('--chunk_size', type=int, default=4096)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = TensorVMSplit(**kwargs)
    tensorf.load(ckpt)
    tensorf.eval()

    # 加载测试数据
    dataset = dataset_dict['blender']
    test_dataset = dataset(args.datadir, split='test', downsample=1.0, is_stack=True)

    # 输出路径
    savePath = os.path.join('./renders', args.expname)
    os.makedirs(savePath, exist_ok=True)

    # 执行渲染
    evaluation_feature(
        test_dataset=test_dataset,
        tensorf=tensorf,
        args=None,
        renderer=OctreeRender_trilinear_fast,
        chunk_size=args.chunk_size,
        savePath=savePath,
        N_vis=args.N_vis,
        white_bg=False,
        ndc_ray=False,
        style_img=None,
        device=device
    )

    print(f"\n✅ Rendering complete! Results saved to: {savePath}")
