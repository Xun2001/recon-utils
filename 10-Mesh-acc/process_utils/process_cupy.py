import numpy as np
from PIL import Image
import cupy as cp
from cupyx.scipy.spatial import Delaunay
from cupyx.scipy.interpolate import LinearNDInterpolator

import time
import os

def save_gray(img, title, save_path, max_depth):
    os.makedirs(save_path, exist_ok=True)
    # 检测并替换 NaN 值为 0
    img = np.nan_to_num(img, nan=0)
    img = (img / max_depth * 255).astype('uint8')
    save_file_path = os.path.join(save_path, title + '.png')
    cv2.imwrite(save_file_path, img)


def process_image_stack_cupy_points_time(args, image_path, save_path, file_name):
    """
    use cupy to replace numpy
    """
    depth_bins = args.depth_bins
    num_bins = args.num_bins
    
    MAX_DEPTH = args.max_depth
    MAX_EDGE = args.max_edge
    SHOW = args.show
    
    time_stats = {} # time stats
    start_time = time.time()
    
    image = Image.open(image_path)
    depth_map = (np.array(image) / 255.0 * MAX_DEPTH).astype(np.float32)
    H, W = depth_map.shape
    mask = np.zeros((H, W), dtype=bool)
    final_depth = np.full_like(depth_map, np.nan)
    prev_points = np.array([]).reshape(0, 2)
    prev_depths = np.array([])
    
    time_stats['image_reading'] = time.time() - start_time
    
    # cp.asarray(numpy_array)
    
    for i in range(num_bins):
        bin_start_time = time.time()
        low, high = depth_bins[i], depth_bins[i + 1]
        valid_mask = (depth_map >= low) & (depth_map < high) & (~mask)
        if not np.any(valid_mask):
            # print(f"跳过空区间: {low:.1f}-{high:.1f}m")
            continue
        u, v = np.where(valid_mask) # 行索引和列索引
        depths = depth_map[valid_mask]
        points = np.column_stack((v, u))
        
        all_points = np.vstack((prev_points, points))
        all_depths = np.hstack((prev_depths, depths))

        prev_mask = (depth_map >= (high-0.2)) & (depth_map < high) & (~mask)
        prev_u,prev_v = np.where(prev_mask)
        prev_depths = depth_map[prev_mask]
        prev_points = np.column_stack((prev_v,prev_u))
        
        if SHOW:
            pre_triangulation = np.full((H, W), np.nan, dtype=np.float32)
            pre_triangulation[u, v] = depths
            save_gray(pre_triangulation, f'Bin-{i}-Sparse-Depth', save_path, MAX_DEPTH)
        
        
        try:
            delaunay_start = time.time()
            all_points_cp = cp.asarray(all_points)
            tri_cp = Delaunay(all_points_cp)
            
        except:
            print("Delaunay 剖分失败")
            continue
        delaunay_time = time.time() - delaunay_start
        time_stats[f'bin_{i}_delaunay'] = delaunay_time
        
        edge_filter_start = time.time()
        
        simplices = tri_cp.simplices
        if len(simplices) == 0:
            continue
        pts_cp = all_points_cp[simplices]
        edges_cp = pts_cp - pts_cp[:, [1, 2, 0], :]  # 计算三条边向量
        edge_lengths_cp = np.linalg.norm(edges_cp, axis=2)
        max_edge_lengths_cp = np.max(edge_lengths_cp, axis=1)
        valid_tris_mask_cp = max_edge_lengths_cp <= MAX_EDGE
        valid_tris = simplices[valid_tris_mask_cp]
        if valid_tris.shape[0] == 0:
            continue
        time_new = time.time()

        all_depths_cp = cp.asarray(all_depths)

        unique_indices = cp.unique(valid_tris.flatten())  # shape [K,], K ≤ N
        edge_mask = cp.zeros(all_points_cp.shape[0], dtype=bool)
        edge_mask[unique_indices] = True
        edge_filter_points_cp = all_points_cp[edge_mask]
        edge_filter_depths_cp = all_depths_cp[edge_mask]

        tri_edge = Delaunay(edge_filter_points_cp)
        interp = LinearNDInterpolator(tri_edge, edge_filter_depths_cp, fill_value=cp.nan)
        print(f"time:{time.time()-time_new}")
        grid_x, grid_y = cp.meshgrid(cp.arange(W), cp.arange(H))
        q = cp.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
        z = interp(q)
        img_interp = cp.asnumpy(z).reshape(H, W)
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

