import os
import time

import numpy as np
import cupy as cp
from cupyx.scipy.spatial import Delaunay
# from scipy.spatial import Delaunay
# from matplotlib.tri import Triangulation
from cupyx.scipy.interpolate import LinearNDInterpolator
from PIL import Image                 # pillow-simd 加速图像解码
import torch
from torch.utils.data import Dataset, DataLoader


def process_depth_map(args, depth_map, save_path, file_name):
    depth_bins = args.depth_bins
    H, W = depth_map.shape
    mask = np.zeros((H, W), dtype=bool)
    final_depth = np.full((H, W), np.nan, dtype=np.float32)
    prev_pts = np.empty((0,2), np.float32) # prev_points
    prev_ds  = np.empty((0,), np.float32) # prev_depths

    for i in range(len(depth_bins)-1):
        low, high = depth_bins[i], depth_bins[i+1]
        valid_mask = (depth_map>=low)&(depth_map<high)&(~mask) # valid mask
        if not valid_mask.any():
            continue

        # 当前 bin 的点
        u, v = np.where(valid_mask)
        ds = depth_map[valid_mask].astype(np.float32)
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
        
        try:
            start_tri = time.time()
            tri_gpu = Delaunay(all_pts)
            print(f"Delaunay time: {time.time() - start_tri:.2f}s")
        except:
            print(f"Bin {i} Delaunay triangulation failed")
            continue

        # 边长过滤
        simp = tri_gpu.simplices
        if len(simp) == 0:
            continue
        vs = all_pts[simp]  # (n_tri,3,2)
        edges = vs - vs[:,[1,2,0],:]
        lengths = cp.linalg.norm(edges, axis=2)
        keep_mask = cp.max(lengths, axis=1) <= args.max_edge
        valid_simp = simp[keep_mask]
        if valid_simp.shape[0] == 0:
            # print("No valid simplices")
            continue
        # tri_gpu.simplices = valid_simp
        
        # GPU 插值
        tri_mpl = Triangulation(all_pts[:, 0], all_pts[:, 1], triangles=valid_simp)
        interp = LinearNDInterpolator(tri_mpl, all_ds)
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        q = cp.stack([cp.asarray(grid_x).ravel(), cp.asarray(grid_y).ravel()], axis=1)
        z = interp(q)
        img_interp = cp.asnumpy(z).reshape(H, W)

        new_region = (~np.isnan(img_interp)) & (~mask)
        final_depth[new_region] = img_interp[new_region]
        mask |= new_region

    # 空洞填充为原图深度
    out = np.where(mask, final_depth, depth_map)
    # 保存结果
    os.makedirs(save_path, exist_ok=True)
    out_img = (out / args.max_depth * 255.0).astype(np.uint8)
    Image.fromarray(out_img).save(os.path.join(save_path, f"{file_name}_out.png"))
