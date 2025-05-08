import os
import glob
import argparse
import time

import numpy as np
import cupy as cp
from cupyx.scipy.spatial import Delaunay
from cupyx.scipy.interpolate import LinearNDInterpolator
from PIL import Image                 # pillow-simd 加速图像解码
import torch
from torch.utils.data import Dataset, DataLoader

from func_utils.config import *


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

def process_depth_map(args, depth_map, save_path, file_name):
    """
    在 GPU 上对单张深度图做分箱、Delaunay 三角化 + 线性插值 densification
    """
    depth_bins = args.depth_bins
    H, W = depth_map.shape
    mask = np.zeros((H, W), dtype=bool)
    final_depth = np.full((H, W), np.nan, dtype=np.float32)
    prev_pts = np.empty((0,2), np.float32) # prev_points
    prev_ds  = np.empty((0,), np.float32) # prev_depths

    for i in range(len(depth_bins)-1):
        low, high = depth_bins[i], depth_bins[i+1]
        valid = (depth_map>=low)&(depth_map<high)&(~mask) # valid mask
        if not valid.any():
            continue

        # 当前 bin 的点
        u, v = np.nonzero(valid)
        ds = depth_map[valid].astype(np.float32)
        pts = np.stack([v, u], axis=1).astype(np.float32)

        # 合并点与深度
        all_pts = np.vstack([prev_pts, pts])
        all_ds  = np.hstack([prev_ds, ds])

        # 更新 prev_mask prev_depths 
        ## 前一个 bin 内 0.5m 范围内的点
        pmask = (depth_map>=(high-0.5))&(depth_map<high)&(~mask)
        pu,pv = np.nonzero(pmask)
        prev_pts = np.stack([pv, pu], axis=1).astype(np.float32)
        prev_ds  = depth_map[pmask].astype(np.float32)
        
        # GPU 三角化
        pts_gpu = cp.asarray(all_pts)
        d_gpu   = cp.asarray(all_ds) # depths
        try:
            start_tri = time.time()
            tri_gpu = Delaunay(pts_gpu)
            print(f"Delaunay time: {time.time() - start_tri:.2f}s")
        except:
            print(f"Bin {i} Delaunay triangulation failed")
            continue

        # 边长过滤
        simp = tri_gpu.simplices
        vs = pts_gpu[simp]  # (n_tri,3,2)
        edges = vs - vs[:,[1,2,0],:]
        lengths = cp.linalg.norm(edges, axis=2)
        keep_mask = cp.max(lengths, axis=1) <= args.max_edge
        valid_simp = simp[keep_mask]
        if valid_simp.shape[0] == 0:
            continue
        tri_gpu.simplices = valid_simp
        assert valid_simp.dtype == tri_gpu.simplices.dtype
        assert int(valid_simp.max()) < pts_gpu.shape[0]

        
        # GPU 插值
        interp = LinearNDInterpolator(tri_gpu, d_gpu, fill_value=cp.nan)
        # grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        grid_x, grid_y = cp.meshgrid(cp.arange(W), cp.arange(H))
        q = cp.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
        z = interp(q)
        img_interp = cp.asnumpy(z).reshape(H, W)
        args.show = True
        if args.show:
            interp_img = (img_interp / args.max_depth * 255.0).astype(np.uint8)
            interp_file_name = f"{file_name}_bin_{i}_interp.png"
            Image.fromarray(interp_img).save(os.path.join(save_path, interp_file_name))
        
        new_region = (~np.isnan(img_interp)) & (~mask)
        final_depth[new_region] = img_interp[new_region]
        mask |= new_region

    # 空洞填充为原图深度
    out = np.where(mask, final_depth, depth_map)
    # 保存结果
    os.makedirs(save_path, exist_ok=True)
    out_img = (out / args.max_depth * 255.0).astype(np.uint8)
    Image.fromarray(out_img).save(os.path.join(save_path, f"{file_name}_out.png"))

def main():
    args = mesh_config()
    args.batch_size = 10
    # DataLoader 并行加载
    dataset = DepthDataset(args.input_folder, args.max_depth)
    loader  = DataLoader(dataset,
                         batch_size=args.batch_size,
                         shuffle=False,
                         num_workers=args.num_workers,
                         pin_memory=True)

    # 遍历每个批次
    for batch in loader:
        depths, names = batch
        # 每张图单独处理
        for depth_map, fname in zip(depths, names):
            depth_map = depth_map.numpy()  # Tensor -> NumPy
            process_depth_map(args, depth_map,
                              args.output_folder, fname)

if __name__ == "__main__":
    main()
