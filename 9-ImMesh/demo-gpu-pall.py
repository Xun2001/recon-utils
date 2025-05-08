import os
import argparse
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import cupy as cp
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2

from func_utils.config import *

def plt_show_gray(img, title, save_path, max_depth):
    img = (img / max_depth * 255).astype('uint8')
    save_file_path = os.path.join(save_path, title + '.png')
    cv2.imwrite(save_file_path, img)

class DepthDataset(Dataset):
    def __init__(self, image_dir, max_depth):
        self.image_dir = image_dir
        self.files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        self.max_depth = max_depth

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        path = os.path.join(self.image_dir, file_name)
        img = Image.open(path)
        depth_map = np.array(img, dtype=np.float32) / 255.0 * self.max_depth
        depth_tensor = torch.from_numpy(depth_map)  # [H, W]
        return depth_tensor, file_name

def cupy_barycentric_interpolation(tris_np, pts_np, depths_np, H, W):
    pts_cp = cp.asarray(pts_np)
    depths_cp = cp.asarray(depths_np)
    grid_y, grid_x = cp.meshgrid(cp.arange(H), cp.arange(W))
    flat_xy = cp.stack((grid_x.ravel(), grid_y.ravel()), axis=1)
    interp_vals = cp.full((H * W,), cp.nan, dtype=cp.float32)

    for tri in tris_np:
        i0, i1, i2 = tri
        v0, v1, v2 = pts_cp[i0], pts_cp[i1], pts_cp[i2]
        d0, d1, d2 = depths_cp[i0], depths_cp[i1], depths_cp[i2]

        denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
        if cp.abs(denom) < 1e-6:
            continue

        w0 = ((v1[1] - v2[1]) * (flat_xy[:, 0] - v2[0]) + (v2[0] - v1[0]) * (flat_xy[:, 1] - v2[1])) / denom
        w1 = ((v2[1] - v0[1]) * (flat_xy[:, 0] - v2[0]) + (v0[0] - v2[0]) * (flat_xy[:, 1] - v2[1])) / denom
        w2 = 1 - w0 - w1

        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not cp.any(inside):
            continue

        interp = w0 * d0 + w1 * d1 + w2 * d2
        flat_idx = flat_xy[inside][:, 1] * W + flat_xy[inside][:, 0]
        interp_vals[flat_idx] = interp[inside]

    return cp.asnumpy(interp_vals.reshape(H, W))

def interpolate_bin_parallel(bin_index, depth_map_np, depth_bins, mask_np, prev_points_np, prev_depths_np, H, W, max_edge):
    low, high = depth_bins[bin_index], depth_bins[bin_index + 1]
    valid_mask = (depth_map_np >= low) & (depth_map_np < high) & (~mask_np)
    if not np.any(valid_mask):
        return bin_index, None

    u, v = np.where(valid_mask)
    depths = depth_map_np[u, v]
    points = np.column_stack((v, u))

    all_points = np.vstack((prev_points_np, points))
    all_depths = np.hstack((prev_depths_np, depths))

    try:
        tri = Delaunay(all_points)
    except:
        return bin_index, None

    simplices = tri.simplices
    if simplices.shape[0] == 0:
        return bin_index, None

    tri_pts = all_points[simplices]
    edges = tri_pts - tri_pts[:, [1,2,0], :]
    lengths = np.linalg.norm(edges, axis=2)
    max_len = np.max(lengths, axis=1)
    valid_tris = simplices[max_len <= max_edge]
    if valid_tris.shape[0] == 0:
        return bin_index, None

    interp_image_np = cupy_barycentric_interpolation(valid_tris, all_points, all_depths, H, W)
    return bin_index, interp_image_np

def process_image_stack_acc_points_torch(depth_map, args, file_name, save_path):
    depth_bins = args.depth_bins
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    depth_map = depth_map.to(device)
    H, W = depth_map.shape
    final_depth = torch.full((H, W), float('nan'), device=device)
    mask_np = np.zeros((H, W), dtype=bool)
    depth_map_np = depth_map.cpu().numpy()

    MAX_EDGE = args.max_edge
    SHOW = args.show
    MAX_DEPTH = args.max_depth

    # 初始化上一bin的信息
    prev_points_np = np.empty((0, 2), dtype=np.int32)
    prev_depths_np = np.empty((0,), dtype=np.float32)

    results = {}
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(interpolate_bin_parallel, i, depth_map_np, depth_bins, mask_np,
                            prev_points_np, prev_depths_np, H, W, MAX_EDGE): i
            for i in range(len(depth_bins) - 1)
        }
        for future in as_completed(futures):
            idx, interp = future.result()
            results[idx] = interp

    for i in range(len(depth_bins) - 1):
        interp = results.get(i)
        if interp is None:
            continue
        interp_torch = torch.from_numpy(interp).to(device)
        new_region = (~torch.isnan(interp_torch)) & (~torch.tensor(mask_np, device=device))
        final_depth[new_region] = interp_torch[new_region]
        mask_np |= new_region.cpu().numpy()

        if SHOW:
            plt_show_gray(final_depth.cpu().numpy(), f'Bin-{i}-Inter', save_path, MAX_DEPTH)

    filled = torch.where(torch.tensor(mask_np, device=device), final_depth, depth_map)
    plt_show_gray(filled.cpu().numpy(), f"{file_name}-final", save_path, MAX_DEPTH)

def main():
    args = mesh_config()
    args.batch_size = 4

    save_path = os.path.join(args.output_folder,args.process_fun)
    dataset = DepthDataset(args.input_folder, args.max_depth)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    for depth_batch, file_names in loader:
        for depth_map, fname in zip(depth_batch, file_names):
            process_image_stack_acc_points_torch(depth_map, args, fname, save_path)

if __name__ == '__main__':
    main()
