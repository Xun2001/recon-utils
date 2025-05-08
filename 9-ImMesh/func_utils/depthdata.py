import os
import argparse
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation


def plt_show_gray(img, title, save_path, max_depth):
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray', vmin=0, vmax=max_depth)
    plt.title(title)
    plt.axis('off')
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, title + '.png'))
    plt.close()


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

def torch_barycentric_interpolation(tris, pts_t, depths_t, H, W, device):
    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device))
    interp_vals = torch.full((H * W,), float('nan'), device=device)

    for tri_idx in range(tris.shape[0]):
        idx0, idx1, idx2 = tris[tri_idx]
        v0 = pts_t[idx0]
        v1 = pts_t[idx1]
        v2 = pts_t[idx2]
        d0 = depths_t[idx0]
        d1 = depths_t[idx1]
        d2 = depths_t[idx2]

        denom = ((v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1]))
        if abs(denom) < 1e-6:
            continue

        min_x = int(max(min(v0[0], v1[0], v2[0]).item(), 0))
        max_x = int(min(max(v0[0], v1[0], v2[0]).item(), W - 1))
        min_y = int(max(min(v0[1], v1[1], v2[1]).item(), 0))
        max_y = int(min(max(v0[1], v1[1], v2[1]).item(), H - 1))

        ys = torch.arange(min_y, max_y + 1, device=device)
        xs = torch.arange(min_x, max_x + 1, device=device)
        yy, xx = torch.meshgrid(ys, xs)
        pts_tile = torch.stack((xx.flatten(), yy.flatten()), dim=1)

        w0 = ((v1[1] - v2[1]) * (pts_tile[:, 0] - v2[0]) + (v2[0] - v1[0]) * (pts_tile[:, 1] - v2[1])) / denom
        w1 = ((v2[1] - v0[1]) * (pts_tile[:, 0] - v2[0]) + (v0[0] - v2[0]) * (pts_tile[:, 1] - v2[1])) / denom
        w2 = 1 - w0 - w1

        mask_in = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not mask_in.any():
            continue

        depths_tri = w0[mask_in] * d0 + w1[mask_in] * d1 + w2[mask_in] * d2
        pts_inside = pts_tile[mask_in].long()
        idxs = pts_inside[:, 1] * W + pts_inside[:, 0]

        interp_vals[idxs] = depths_tri

    return interp_vals.reshape(H, W)

def process_image_stack_acc_points_torch(depth_map, depth_bins, args, file_name, save_path):
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

        # current bin points
        coords = torch.nonzero(valid_mask, as_tuple=False)  # [N,2] (u, v)
        u, v = coords[:, 0], coords[:, 1]
        depths = depth_map[u, v]
        points = torch.stack((v, u), dim=1)

        # accumulate points
        all_points = torch.cat((prev_points, points), dim=0)
        all_depths = torch.cat((prev_depths, depths), dim=0)

        # save for next bin
        prev_mask = (depth_map >= (high - 0.5)) & (depth_map < high) & (~mask)
        prev_coords = torch.nonzero(prev_mask, as_tuple=False)
        prev_points = torch.stack((prev_coords[:, 1], prev_coords[:, 0]), dim=1)
        prev_depths = depth_map[prev_coords[:, 0], prev_coords[:, 1]]

        if SHOW:
            pre_tri = torch.full((H, W), float('nan'), device=device)
            pre_tri[u, v] = depths
            plt_show_gray(pre_tri.cpu().numpy(), f'Bin-{i}-Sparse-Depth', save_path, MAX_DEPTH)

        # Delaunay on CPU
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

        # filter edges
        tri_pts = pts_np[simplices]  # [T,3,2]
        edges = tri_pts - tri_pts[:, [1,2,0], :]
        lengths = np.linalg.norm(edges, axis=2)
        max_len = np.max(lengths, axis=1)
        valid_tris = simplices[max_len <= MAX_EDGE]
        if valid_tris.shape[0] == 0:
            continue

        # convert for torch interpolation
        tris = torch.from_numpy(valid_tris).long().to(device)
        pts_t = all_points.to(device).float()  # [M,2]
        depths_t = all_depths.to(device)

        # voxelize per-triangle interpolation
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device))
        points_xy = torch.stack((grid_x.flatten(), grid_y.flatten()), dim=1).float()  # [HW,2]

        interp_vals = torch.full((H * W,), float('nan'), device=device)

        # naive loop over triangles (could be optimized)
        for tri_idx in range(tris.shape[0]):
            idx0, idx1, idx2 = tris[tri_idx]
            v0 = pts_t[idx0]
            v1 = pts_t[idx1]
            v2 = pts_t[idx2]
            d0 = depths_t[idx0]
            d1 = depths_t[idx1]
            d2 = depths_t[idx2]

            # barycentric denominator
            denom = ((v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1]))
            if abs(denom) < 1e-6:
                continue

            # bounding box
            min_x = int(max(min(v0[0], v1[0], v2[0]).item(), 0))
            max_x = int(min(max(v0[0], v1[0], v2[0]).item(), W-1))
            min_y = int(max(min(v0[1], v1[1], v2[1]).item(), 0))
            max_y = int(min(max(v0[1], v1[1], v2[1]).item(), H-1))

            # grid within bbox
            ys = torch.arange(min_y, max_y+1, device=device)
            xs = torch.arange(min_x, max_x+1, device=device)
            yy, xx = torch.meshgrid(ys, xs)
            pts_tile = torch.stack((xx.flatten(), yy.flatten()), dim=1)

            # compute barycentric weights
            w0 = ((v1[1] - v2[1]) * (pts_tile[:,0] - v2[0]) + (v2[0] - v1[0]) * (pts_tile[:,1] - v2[1])) / denom
            w1 = ((v2[1] - v0[1]) * (pts_tile[:,0] - v2[0]) + (v0[0] - v2[0]) * (pts_tile[:,1] - v2[1])) / denom
            w2 = 1 - w0 - w1

            mask_in = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            if not mask_in.any():
                continue

            # interpolated depths
            depths_tri = w0[mask_in] * d0 + w1[mask_in] * d1 + w2[mask_in] * d2
            pts_inside = pts_tile[mask_in].long()
            idxs = pts_inside[:,1] * W + pts_inside[:,0]

            # assign
            interp_vals[idxs] = depths_tri

        interp_image = interp_vals.reshape(H, W)
        new_region = (~torch.isnan(interp_image)) & (~mask)
        final_depth[new_region] = interp_image[new_region]
        mask |= new_region

        if SHOW:
            plt_show_gray(final_depth.cpu().numpy(), f'Bin-{i}-Inter', save_path, MAX_DEPTH)

    # fill remaining
    filled = torch.where(mask, final_depth, depth_map)
    plt_show_gray(filled.cpu().numpy(), f"{file_name}-final", save_path, MAX_DEPTH)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--max_depth', type=float, default=100.0)
    parser.add_argument('--max_edge', type=float, default=1.0)
    parser.add_argument('--bins', type=int, default=10)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    # create bins
    depth_bins = np.linspace(0, args.max_depth, args.bins + 1)

    dataset = DepthDataset(args.image_dir, args.max_depth)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    for depth_batch, file_names in loader:
        for depth_map, fname in zip(depth_batch, file_names):
            process_image_stack_acc_points_torch(depth_map, depth_bins, args, fname, args.save_path)

if __name__ == '__main__':
    main()
