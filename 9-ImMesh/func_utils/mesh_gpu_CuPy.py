import os
import time
import json
import cupy as cp
import numpy as np
from PIL import Image
from cupyx.scipy.spatial import Delaunay
from cupyx.scipy.interpolate import LinearNDInterpolator

def process_image_gpu(args, image_path, save_path):
    MAX_DEPTH = args.max_depth
    MAX_EDGE = args.max_edge
    depth_bins = args.depth_bins
    num_bins = args.num_bins
    
    # SHOW = args.show # debug
    file_name = os.path.splitext(os.path.basename(image_path))[0]
    time_stats = {}
    
    start_time = time.time()

    # 读取图像并转换为深度图
    image = Image.open(image_path).convert('L')
    depth_map = (np.array(image) / 255.0 * MAX_DEPTH).astype(np.float32)
    H, W = depth_map.shape
    depth_map_gpu = cp.asarray(depth_map)
    mask_gpu = cp.zeros((H, W), dtype=cp.bool_)
    final_depth_gpu = cp.full_like(depth_map_gpu, cp.nan)
    prev_points_gpu = cp.empty((0, 2), dtype=cp.float32)
    prev_depths_gpu = cp.empty((0,), dtype=cp.float32)

    time_stats['image_reading'] = time.time() - start_time

    for i in range(num_bins):
        bin_start_time = time.time()
        low, high = depth_bins[i], depth_bins[i + 1]
        valid_mask_gpu = (depth_map_gpu >= low) & (depth_map_gpu < high) & (~mask_gpu)
        if not cp.any(valid_mask_gpu):
            continue

        u_gpu, v_gpu = cp.where(valid_mask_gpu)
        depths_gpu = depth_map_gpu[valid_mask_gpu]
        points_gpu = cp.stack((v_gpu, u_gpu), axis=1)

        all_points_gpu = cp.concatenate((prev_points_gpu, points_gpu), axis=0)
        all_depths_gpu = cp.concatenate((prev_depths_gpu, depths_gpu), axis=0)

        prev_mask_gpu = (depth_map_gpu >= (high - 0.5)) & (depth_map_gpu < high) & (~mask_gpu)
        prev_u_gpu, prev_v_gpu = cp.where(prev_mask_gpu)
        prev_depths_gpu = depth_map_gpu[prev_mask_gpu]
        prev_points_gpu = cp.stack((prev_v_gpu, prev_u_gpu), axis=1)

        try:
            delaunay_start = time.time()
            tri = Delaunay(all_points_gpu)
            delaunay_time = time.time() - delaunay_start
            time_stats[f'bin_{i}_delaunay'] = delaunay_time
        except:
            print(f"Delaunay triangulation failed for bin {i}")
            continue

        edge_filter_start = time.time()
        simplices_gpu = tri.simplices
        if simplices_gpu.shape[0] == 0:
            continue
        pts_gpu = all_points_gpu[simplices_gpu]
        edges_gpu = pts_gpu - pts_gpu[:, [1, 2, 0], :]
        edge_lengths_gpu = cp.linalg.norm(edges_gpu, axis=2)
        max_edge_lengths_gpu = cp.max(edge_lengths_gpu, axis=1)
        valid_tris_mask_gpu = max_edge_lengths_gpu <= MAX_EDGE
        valid_tris_gpu = simplices_gpu[valid_tris_mask_gpu]
        
        time_stats[f'bin_{i}_edge_filter'] = time.time() - edge_filter_start

        if valid_tris_gpu.shape[0] == 0:
            continue
        # all_points_gpu.simplices = valid_tris_gpu
        # interp = LinearNDInterpolator(tri_gpu, d_gpu, fill_value=cp.nan)
        # valid_points_gpu = all_points_gpu[valid_tris_gpu.flatten()]

        interpolation_start = time.time()
        tri.simplices = valid_tris_gpu
        
        interpolator = LinearNDInterpolator(tri, all_depths_gpu)
        grid_v_gpu, grid_u_gpu = cp.meshgrid(cp.arange(W), cp.arange(H))
        grid_points_gpu = cp.stack((grid_v_gpu.ravel(), grid_u_gpu.ravel()), axis=-1)
        interpolated_gpu = interpolator(grid_points_gpu)
        interp_image_gpu = interpolated_gpu.reshape((H, W))
        new_region_gpu = (~cp.isnan(interp_image_gpu)) & (~mask_gpu)
        final_depth_gpu = cp.where(new_region_gpu, interp_image_gpu, final_depth_gpu)
        mask_gpu = mask_gpu | new_region_gpu
        interpolation_time = time.time() - interpolation_start
        time_stats[f'bin_{i}_interpolation'] = interpolation_time
        time_stats[f'bin_{i}_total'] = time.time() - bin_start_time

    final_start_time = time.time()
    final_depth_gpu = cp.where(mask_gpu, final_depth_gpu, depth_map_gpu)
    time_stats['final_processing'] = time.time() - final_start_time
    
    save_img_time = time.time()
    final_depth_cpu = cp.asnumpy(final_depth_gpu)
    output_image = Image.fromarray((final_depth_cpu / MAX_DEPTH * 255).astype(np.uint8))
    output_image.save(os.path.join(save_path, f"{file_name}_final.png"))
    time_stats['save_image'] = time.time() - save_img_time

    # 保存时间统计到 JSON 文件
    json_path = os.path.join(save_path, f'{file_name}_time_stats.json')
    with open(json_path, 'w') as f:
        json.dump(time_stats, f, indent=4)

