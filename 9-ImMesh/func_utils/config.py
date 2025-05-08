import argparse
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import glob
from PIL import Image
def mesh_config():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='深度图插值脚本')
    parser.add_argument('--input_folder',"-i", type=str, default="/home/qinllgroup/hongxiangyu/git_project/livo2-data-utils/9-ImMesh/data/demo/images")
    parser.add_argument('--output_folder',"-o", type=str, default="/home/qinllgroup/hongxiangyu/git_project/livo2-data-utils/9-ImMesh/data/demo/debug")
    parser.add_argument('--max_depth', type=float, default=30, help='最大深度')
    parser.add_argument('--max_edge', type=float, default=10, help='最大允许三角形边长（像素）')
    parser.add_argument('--show', action='store_true', help='是否保存中间结果')
    parser.add_argument('--bin_interval', type=float, default=2, help='分 BIN 的间距')
    parser.add_argument('--process_fun', type=str, default='stack_gpu_single', help='bin or stack')
    parser.add_argument('--num_workers', type=int, default=4, help='最大线程数')
    args = parser.parse_args()
    
    # 参数配置
    args.depth_bins = np.arange(0.01, args.max_depth + 0.51, args.bin_interval)
    args.num_bins = len(args.depth_bins) - 1
    
    # 输出当前主要参数
    print(f"参数配置：")
    print(f"最大深度：{args.max_depth}")
    print(f"最大允许三角形边长（像素）：{args.max_edge}")
    print(f"分 BIN 的间距：{args.bin_interval}")
    print(f"处理方式：{args.process_fun}")

    return args


class DepthDataset(Dataset):
    """自定义深度图数据集：读取 PNG、归一化并转换为 float32 数组"""
    def __init__(self, folder, max_depth):
        self.files = glob.glob(os.path.join(folder, "*.png"), recursive=True)
        self.max_depth = max_depth

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path)  # pillow-simd 解码
        depth = (np.array(img, dtype=np.float32) / 255.0) * self.max_depth
        fname = os.path.basename(path)
        return depth, fname