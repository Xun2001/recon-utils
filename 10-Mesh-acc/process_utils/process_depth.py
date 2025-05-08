import os
import glob
import cv2
import numpy as np
import cupy as cp
import os
import numpy as np
from PIL import Image
from scipy.spatial import Delaunay
from matplotlib.tri import LinearTriInterpolator, Triangulation
import matplotlib.pyplot as plt

import json
import time


def save_gray(img, title, save_path, max_depth):
    os.makedirs(save_path, exist_ok=True)
    # 检测并替换 NaN 值为 0
    img = np.nan_to_num(img, nan=0)
    img = (img / max_depth * 255).astype('uint8')
    save_file_path = os.path.join(save_path, title + '.png')
    cv2.imwrite(save_file_path, img)


def process_single_image(args, depth_bins, num_bins, input_folder, output_folder, png_file):
    image_path = os.path.join(input_folder, png_file)

    # 调用原始处理函数
    process_depth_map_single(
        args,
        depth_bins,
        num_bins,
        image_path,
        output_folder,
        os.path.splitext(png_file)[0]
    )

def process_depth_map_single(args,depth_bins,num_bins, image_path, save_path, file_name):
    """
    from process_image_stack_acc_points func in 9-ImMesh/func_utils/mesh_cpu.py
    处理一张深度图，使用 Delaunay 剖分和三角插值进行深度插值。
    只增加前一个 bin 中 0.2m 距离内的点

    """
    MAX_DEPTH = args.max_depth
    MAX_EDGE = args.max_edge
    SHOW = args.show
    
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    depth_map = (np.array(image) / 255.0 * MAX_DEPTH).astype(np.float32)
    H, W = depth_map.shape
    mask = np.zeros((H, W), dtype=bool)
    final_depth = np.full_like(depth_map, np.nan)
    prev_points = np.array([]).reshape(0, 2)
    prev_depths = np.array([])

    for i in range(num_bins):
        low, high = depth_bins[i], depth_bins[i + 1]
        valid_mask = (depth_map >= low) & (depth_map < high) & (~mask)
        if not np.any(valid_mask):
            # print(f"跳过空区间: {low:.1f}-{high:.1f}m")
            continue
        
        u, v = np.where(valid_mask)
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
            tri = Delaunay(all_points)
        except:
            print("Delaunay 剖分失败")
            continue

        simplices = tri.simplices
        if len(simplices) == 0:
            continue
        pts = all_points[simplices]
        edges = pts - pts[:, [1, 2, 0], :]  # 计算三条边向量
        edge_lengths = np.linalg.norm(edges, axis=2)
        max_edge_lengths = np.max(edge_lengths, axis=1)
        valid_tris_mask = max_edge_lengths <= MAX_EDGE
        valid_tris = simplices[valid_tris_mask]

        if len(valid_tris) == 0:
            # print("无有效三角形")
            continue

        tri_mpl = Triangulation(all_points[:, 0], all_points[:, 1], triangles=valid_tris)
        interpolator = LinearTriInterpolator(tri_mpl, all_depths)
        grid_v, grid_u = np.meshgrid(np.arange(W), np.arange(H))
        interpolated = interpolator(grid_v.ravel(), grid_u.ravel())
        interp_image = interpolated.reshape((H, W))
        new_region = (~np.isnan(interp_image)) & (~mask)
        final_depth[new_region] = interp_image[new_region]
        mask |= new_region

        if SHOW:
            save_gray(interp_image, f'Bin-{i}-Inter', save_path, MAX_DEPTH)


    final_depth = np.where(mask, final_depth, depth_map)
    save_gray(final_depth, f"{file_name}-final", save_path, MAX_DEPTH)
    


def process_image_stack_acc_points_time(args, image_path, save_path, file_name):
    """
    处理一张深度图，使用 Delaunay 剖分和三角插值进行深度插值。
    只增加前一个 bin 中 0.2m 距离内的点

    acc: use numpy replace for in edge filter
    Args:
        image_path (_type_): _description_
        save_path (_type_): _description_
        file_name (_type_): _description_
    """
    depth_bins = args.depth_bins
    num_bins = args.num_bins
    
    MAX_DEPTH = args.max_depth
    MAX_EDGE = args.max_edge
    SHOW = args.show
    
    time_stats = {}
    
    start_time = time.time()
    
    image = Image.open(image_path)
    depth_map = (np.array(image) / 255.0 * MAX_DEPTH).astype(np.float32)
    H, W = depth_map.shape
    mask = np.zeros((H, W), dtype=bool)
    final_depth = np.full_like(depth_map, np.nan)
    prev_points = np.array([]).reshape(0, 2)
    prev_depths = np.array([])  
    
    time_stats['image_reading'] = time.time() - start_time
    
    for i in range(num_bins):                
        bin_start_time = time.time()
        
        low, high = depth_bins[i], depth_bins[i + 1]
        valid_mask = (depth_map >= low) & (depth_map < high) & (~mask)
        if not np.any(valid_mask):
            # print(f"跳过空区间: {low:.1f}-{high:.1f}m")
            continue
        
        u, v = np.where(valid_mask)
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
            tri = Delaunay(all_points)
            delaunay_start = time.time()
        except:
            print("Delaunay 剖分失败")
            continue
        
        delaunay_time = time.time() - delaunay_start
        time_stats[f'bin_{i}_delaunay'] = delaunay_time
        
        
        edge_filter_start = time.time()
        
        simplices = tri.simplices
        if len(simplices) == 0:
            continue
        pts = all_points[simplices]
        edges = pts - pts[:, [1, 2, 0], :]  # 计算三条边向量
        edge_lengths = np.linalg.norm(edges, axis=2)
        max_edge_lengths = np.max(edge_lengths, axis=1)
        valid_tris_mask = max_edge_lengths <= MAX_EDGE
        valid_tris = simplices[valid_tris_mask]

        time_stats[f'bin_{i}_edge_filter'] = time.time() - edge_filter_start
        
        if len(valid_tris) == 0:
            # print("无有效三角形")
            continue
        
        
        interpolation_start = time.time()
        
        tri_mpl = Triangulation(all_points[:, 0], all_points[:, 1], triangles=valid_tris)
        interpolator = LinearTriInterpolator(tri_mpl, all_depths)
        grid_v, grid_u = np.meshgrid(np.arange(W), np.arange(H))
        interpolated = interpolator(grid_v.ravel(), grid_u.ravel())
        interp_image = interpolated.reshape((H, W))
        new_region = (~np.isnan(interp_image)) & (~mask)
        final_depth[new_region] = interp_image[new_region]
        mask |= new_region

        interpolation_time = time.time() - interpolation_start
        time_stats[f'bin_{i}_interpolation'] = interpolation_time
        
        if SHOW:
            save_gray(interp_image, f'Bin-{i}-Inter', save_path, MAX_DEPTH)
            
        bin_time = time.time() - bin_start_time
        time_stats[f'bin_{i}_total'] = bin_time

    final_start_time = time.time()
    
    final_depth = np.where(mask, final_depth, depth_map)
    save_gray(final_depth, f"{file_name}-final", save_path, MAX_DEPTH)
    
    time_stats['final_processing'] = time.time() - final_start_time
    time_stats['total_processing'] = time.time() - start_time
    
    # 保存时间统计到 JSON 文件
    json_path = os.path.join(save_path, "time_json")
    os.makedirs(json_path, exist_ok=True)
    json_file = os.path.join(json_path, f'{file_name}_time_stats.json')
    with open(json_file, 'w') as f:
        json.dump(time_stats, f, indent=4)


def process_image_stack_pall_bin_points_time(args, image_path, save_path, file_name):
    """
        在acc的基础上
        进行多bin的并行处理
    """
    depth_bins = args.depth_bins
    num_bins = args.num_bins
    
    MAX_DEPTH = args.max_depth
    MAX_EDGE = args.max_edge
    SHOW = args.show