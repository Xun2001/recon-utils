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
    os.makedirs(save_path, exist_ok=True)
    img = np.nan_to_num(img, nan=0)
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
    tris = tris_np
    pts_cp = cp.asarray(pts_np)
    depths_cp = cp.asarray(depths_np)

    grid_y, grid_x = cp.meshgrid(cp.arange(H), cp.arange(W))
    flat_xy = cp.stack((grid_x.ravel(), grid_y.ravel()), axis=1)
    interp_vals = cp.full((H * W,), cp.nan, dtype=cp.float32)

    for tri in tris:
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

def process_image_stack_acc_points_torch(depth_map, args, file_name, save_path):
    depth_bins = args.depth_bins
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    depth_map = depth_map.to(device)
    H, W = depth_map.shape
    mask = torch.zeros((H, W), dtype=torch.bool, device=device)
    final_depth = torch.full((H, W), float('nan'), device=device)

    prev_points = torch.empty((0, 2), dtype=torch.int64, device=device)
    prev_depths = torch.empty((0,), dtype=torch.float32, device=device)

    MAX_EDGE = args.max_edge
    SHOW = args.show
    MAX_DEPTH = args.max_depth

    for i in range(len(depth_bins) - 1):
        low, high = depth_bins[i], depth_bins[i + 1]
        valid_mask = (depth_map >= low) & (depth_map < high) & (~mask)
        if not valid_mask.any():
            continue

        coords = torch.nonzero(valid_mask, as_tuple=False)
        u, v = coords[:, 0], coords[:, 1]
        depths = depth_map[u, v]
        points = torch.stack((v, u), dim=1)

        all_points = torch.cat((prev_points, points), dim=0)
        all_depths = torch.cat((prev_depths, depths), dim=0)

        prev_mask = (depth_map >= (high - 0.5)) & (depth_map < high) & (~mask)
        prev_coords = torch.nonzero(prev_mask, as_tuple=False)
        prev_points = torch.stack((prev_coords[:, 1], prev_coords[:, 0]), dim=1)
        prev_depths = depth_map[prev_coords[:, 0], prev_coords[:, 1]]

        if SHOW:
            pre_tri = torch.full((H, W), float('nan'), device=device)
            pre_tri[u, v] = depths
            plt_show_gray(pre_tri.cpu().numpy(), f'Bin-{i}-Sparse-Depth', save_path, MAX_DEPTH)

        pts_np = all_points.cpu().numpy()
        depths_np = all_depths.cpu().numpy()
        try:
            tri = Delaunay(pts_np)
        except Exception as e:
            print(f"Delaunay failed: {e}")
            continue

        simplices = tri.simplices
        if simplices.shape[0] == 0:
            continue

        tri_pts = pts_np[simplices]
        edges = tri_pts - tri_pts[:, [1,2,0], :]
        lengths = np.linalg.norm(edges, axis=2)
        max_len = np.max(lengths, axis=1)
        valid_tris = simplices[max_len <= MAX_EDGE]
        if valid_tris.shape[0] == 0:
            continue

        interp_image_np = cupy_barycentric_interpolation(valid_tris, pts_np, depths_np, H, W)
        interp_image = torch.from_numpy(interp_image_np).to(device)

        new_region = (~torch.isnan(interp_image)) & (~mask)
        final_depth[new_region] = interp_image[new_region]
        mask |= new_region

        if SHOW:
            plt_show_gray(final_depth.cpu().numpy(), f'Bin-{i}-Inter', save_path, MAX_DEPTH)

    filled = torch.where(mask, final_depth, depth_map)
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
