import os
import numpy as np
import cupy as cp
from PIL import Image
from cupyx.scipy.spatial import Delaunay
from cupyx.scipy.interpolate import LinearNDInterpolator
from cupy.cuda import Event

def save_gray(arr: np.ndarray, name: str, save_folder: str, max_depth: float):
    """把深度图归一化到 [0,255] 并保存为 PNG。"""
    img = np.nan_to_num(arr, nan=0.0)
    img = (img / max_depth * 255.0).clip(0, 255).astype(np.uint8)
    im = Image.fromarray(img)
    im.save(os.path.join(save_folder, f"{name}.png"))
    
input_folder = "/home/qinllgroup/hongxiangyu/git_project/livo2-data-utils/11-Mesh-gpu/data/tree_01_mini/depth_maps"
output_folder = "/home/qinllgroup/hongxiangyu/git_project/livo2-data-utils/11-Mesh-gpu/data/tree_01_mini/depth_mesh"

num_bins = int(15)
max_edge = 10
max_depth = 30


save_path = os.path.join(output_folder, "stack_gpu") # args.func
json_path = os.path.join(save_path, "time_json")

os.makedirs(save_path, exist_ok=True)
os.makedirs(json_path, exist_ok=True)


for fname in os.listdir(input_folder):
    img = Image.open(os.path.join(input_folder, fname))
    img_arr = (np.array(img) / 255.0 * max_depth).astype(np.float32)
    h, w = img_arr.shape
    depth_map = cp.asarray(img_arr, dtype=cp.float32)
    
    # 1. 提取有效散点
    ys, xs = cp.nonzero(depth_map > 0)
    depths = depth_map[ys, xs]
    pts = cp.stack([xs, ys], axis=1)
    
    # 2.分bin
    overlap = 0.2 # 重叠部分 0.2 m useless
    d_min, d_max = float(depths.min()), float(depths.max())
    depth_bin = cp.linspace(d_min, d_max, num_bins + 1)
    
    final_depth = cp.full_like(depth_map, cp.nan)
    prev_points = cp.array([]).reshape(0, 2)
    prev_depths = cp.array([])
    
    mask = cp.zeros((h, w), dtype=bool)
    for i in range(num_bins):
        low, high = depth_bin[i], depth_bin[i + 1]
        valid_mask = (depth_map >= low) & (depth_map < high) & (~mask)
        if not cp.any(valid_mask):
            continue
        
        u, v = cp.where(valid_mask)
        depths = depth_map[valid_mask]
        points = cp.column_stack((v, u))
        
        all_points = cp.vstack((prev_points, points))
        all_depths = cp.hstack((prev_depths, depths))
        
        prev_mask = (depth_map >= (high-0.2)) & (depth_map < high) & (~mask)
        prev_u,prev_v = cp.where(prev_mask)
        prev_depths = depth_map[prev_mask]
        prev_points = cp.column_stack((prev_v,prev_u))
        
        # 三角剖分
        try:
            tri = Delaunay(all_points)
        except:
            print("Delaunay 剖分失败")
            continue
        simplices = tri.simplices
        pts = all_points[simplices]
        edges = pts - pts[:, [1, 2, 0], :]  
        edge_lengths = cp.linalg.norm(edges, axis=2)
        max_edge_lengths = cp.max(edge_lengths, axis=1)
        valid_tris_mask = max_edge_lengths <= max_edge
        valid_tris = simplices[valid_tris_mask]
        if valid_tris.shape[0] == 0:
            continue
        
        # 线性插值
        interpolator = LinearNDInterpolator(tri, all_depths)
        xs_all, ys_all = cp.meshgrid(cp.arange(w), cp.arange(h))
        query_pts = cp.stack([xs_all.ravel(), ys_all.ravel()], axis=1) # [h*w, 2]
        depth_all = interpolator(query_pts) # non_nan_count = cp.count_nonzero(cp.logical_not(cp.isnan(depth_all)))
        # depth_all = interpolator(xs_all,ys_all)
        
        # simp_idx = tri.find_simplex(query_pts)
        # mask_valid_edge = (simp_idx >= 0)
        
        # false_count = cp.count_nonzero(cp.logical_not(mask_valid_edge))
        # if false_count == (h*w):
        #     print(f"false_count: {false_count}")
        #     continue
        # mask_valid_edge[mask_valid_edge] &= valid_tris[simp_idx[mask_valid_edge]]
        # depth_all[~mask_valid_edge] = cp.nan # 去除插值后无效点处深度
        
        debug = True
        if debug:
            dbg_all = depth_all.reshape(h, w).astype(cp.uint8)
            Image.fromarray(cp.asnumpy(dbg_all)).save(
                os.path.join(output_folder, f'{fname}_bin{i}_interp.png')
            )
        
        new_region = (~cp.isnan(depth_all)) & (~mask)
        final_depth[new_region] = depth_all[new_region]
        mask |= new_region
    
    final_depth = cp.where(mask, final_depth, depth_map)
    
    save_gray(cp.asnumpy(final_depth), f"{fname}-final", save_path, max_depth)