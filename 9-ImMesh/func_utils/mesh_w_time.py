import os
import time
import numpy as np
from scipy.spatial import Delaunay
from matplotlib.tri import Triangulation, LinearTriInterpolator
from PIL import Image
import matplotlib.pyplot as plt
import json

def plt_show_gray(image_arr, title_name: str, save_path, MAX_DEPTH):
    """
    可视化灰度图像并保存为 PNG 文件。
    :param image_arr: 要可视化的图像数组
    :param title_name: 图像的标题，同时也是保存文件的名称
    :param save_path: 保存图像的路径
    """
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_arr = np.nan_to_num(image_arr, nan=0, posinf=0, neginf=0)
        image_arr = (image_arr / MAX_DEPTH * 255).astype(np.uint8)
        image = Image.fromarray(image_arr)
        image_path = os.path.join(save_path, (title_name + '.png'))
        image.save(image_path, 'PNG')

# def process_image_stack_acc_points_time(args, depth_bins, num_bins, image_path, save_path, file_name):
#     """"
#     基于 process_image_stack2_points 进行加速，主要对 边长过滤部分进行加速，同时增加了一些坐标的预计算

#     """
#     MAX_DEPTH = args.max_depth
#     MAX_EDGE = args.max_edge
    
#     timing_file = os.path.join(save_path, (file_name + '_timing.json'))

#     timing_info = {}

#     start_time = time.time()
#     image = Image.open(image_path)
#     depth_map = (np.array(image) / 255.0 * MAX_DEPTH).astype(np.float32)
#     H, W = depth_map.shape
#     mask = np.zeros((H, W), dtype=bool)
#     final_depth = np.full_like(depth_map, np.nan)
#     timing_info["读取图像和初始化"] = time.time() - start_time

#     start_time = time.time()
#     # 预生成网格坐标
#     grid_v, grid_u = np.meshgrid(np.arange(W), np.arange(H))
#     grid_v_flat = grid_v.ravel()
#     grid_u_flat = grid_u.ravel()
#     # 初始化累积点列表
#     prev_points_list = []
#     prev_depths_list = []
#     timing_info["预生成网格坐标和初始化列表"] = time.time() - start_time

#     for i in range(num_bins):
#         print(f"处理第 {i} 个深度区间")
#         start_time = time.time()
#         low, high = depth_bins[i], depth_bins[i + 1]
#         valid_mask = (depth_map >= low) & (depth_map < high) & (~mask)
#         if not np.any(valid_mask):
#             # print(f"跳过空区间: {low:.1f}-{high:.1f}m")
#             continue

#         u, v = np.where(valid_mask)
#         depths = depth_map[valid_mask]
#         current_points = np.column_stack((v.astype(np.float32), u.astype(np.float32)))
#         current_depths = depths.astype(np.float32)
#         timing_info[f"处理第 {i} 个深度区间的有效点"] = time.time() - start_time

#         start_time = time.time()
#         # 合并之前的高区间点和当前点
#         if len(prev_points_list) == 0:
#             all_points = current_points
#             all_depths = current_depths
#         else:
#             all_points = np.vstack([np.array(prev_points_list, dtype=np.float32), current_points])
#             all_depths = np.hstack([np.array(prev_depths_list, dtype=np.float32), current_depths])
#         timing_info[f"合并点和深度（第 {i} 个区间）"] = time.time() - start_time

#         if len(all_points) < 3:
#             continue  # Delaunay需要至少3个点

#         start_time = time.time()
#         # Delaunay三角剖分
#         try:
#             tri = Delaunay(all_points, qhull_options="QJ QbB")  # 优化剖分参数
#         except:
#             print("Delaunay 剖分失败")
#             continue
#         timing_info[f"Delaunay三角剖分（第 {i} 个区间）"] = time.time() - start_time

#         start_time = time.time()
#         # 向量化过滤三角形
#         simplices = tri.simplices
#         if len(simplices) == 0:
#             continue

#         pts = all_points[simplices]
#         edges = pts - pts[:, [1, 2, 0], :]  # 计算三条边向量
#         edge_lengths = np.linalg.norm(edges, axis=2)
#         max_edge_lengths = np.max(edge_lengths, axis=1)
#         valid_tris_mask = max_edge_lengths <= MAX_EDGE
#         valid_tris = simplices[valid_tris_mask]
#         timing_info[f"向量化过滤三角形（第 {i} 个区间）"] = time.time() - start_time

#         if len(valid_tris) == 0:
#             print("无有效三角形")
#             continue

#         start_time = time.time()
#         # 创建插值器
#         tri_mpl = Triangulation(all_points[:, 0], all_points[:, 1], triangles=valid_tris)
#         interpolator = LinearTriInterpolator(tri_mpl, all_depths)
#         timing_info[f"创建插值器（第 {i} 个区间）"] = time.time() - start_time

#         start_time = time.time()
#         # 仅对未处理区域插值
#         unmasked = ~mask
#         interpolated = interpolator(grid_v[unmasked], grid_u[unmasked])
#         valid_interp = ~np.isnan(interpolated)
#         timing_info[f"插值（第 {i} 个区间）"] = time.time() - start_time

#         start_time = time.time()
#         # 更新结果
#         final_depth[unmasked] = np.where(valid_interp, interpolated, final_depth[unmasked])
#         new_mask = unmasked.copy()
#         new_mask[unmasked] = valid_interp
#         mask |= new_mask
#         timing_info[f"更新结果（第 {i} 个区间）"] = time.time() - start_time

#         start_time = time.time()
#         # 更新prev_points为当前bin的高区间点
#         prev_mask = (depth_map >= (high - 0.5)) & (depth_map < high) & (~mask)
#         prev_u, prev_v = np.where(prev_mask)
#         prev_depths = depth_map[prev_mask]
#         prev_points_list = np.column_stack((prev_v, prev_u)).tolist()
#         prev_depths_list = prev_depths.tolist()
#         timing_info[f"更新高区间点（第 {i} 个区间）"] = time.time() - start_time

#     start_time = time.time()
#     # 填充未处理区域为原始值
#     final_depth = np.where(mask, final_depth, depth_map)
#     plt_show_gray(final_depth, f"{file_name}-final", save_path, MAX_DEPTH)
#     timing_info["填充未处理区域和显示图像"] = time.time() - start_time

#     # 将计时信息保存到 JSON 文件
#     with open(timing_file, 'w') as f:
#         json.dump(timing_info, f, indent=4)


def process_image_stack_acc_points_time(args,depth_bins,num_bins, image_path, save_path, file_name):
    """
    处理一张深度图，使用 Delaunay 剖分和三角插值进行深度插值。
    只增加前一个 bin 中 0.5m 距离内的点

    Args:
        image_path (_type_): _description_
        save_path (_type_): _description_
        file_name (_type_): _description_
    """
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
        
        prev_mask = (depth_map >= (high-0.5)) & (depth_map < high) & (~mask)
        prev_u,prev_v = np.where(prev_mask)
        prev_depths = depth_map[prev_mask]
        prev_points = np.column_stack((prev_v,prev_u))

        if SHOW:
            pre_triangulation = np.full((H, W), np.nan, dtype=np.float32)
            pre_triangulation[u, v] = depths
            plt_show_gray(pre_triangulation, f'Bin-{i}-Sparse-Depth', save_path, MAX_DEPTH)
        
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
            plt_show_gray(interp_image, f'Bin-{i}-Inter', save_path, MAX_DEPTH)
            
        bin_time = time.time() - bin_start_time
        time_stats[f'bin_{i}_total'] = bin_time

    final_start_time = time.time()
    
    final_depth = np.where(mask, final_depth, depth_map)
    plt_show_gray(final_depth, f"{file_name}-final", save_path, MAX_DEPTH)
    
    time_stats['final_processing'] = time.time() - final_start_time
    
    # 保存时间统计到 JSON 文件
    json_path = os.path.join(save_path, f'{file_name}_time_stats.json')
    with open(json_path, 'w') as f:
        json.dump(time_stats, f, indent=4)


def process_image_stack_acc_points_delete(args, depth_bins, num_bins, image_path, save_path, file_name):
    """"
    基于 process_image_stack2_points 进行加速，主要对 边长过滤部分进行加速，同时增加了一些坐标的预计算
    
    """
    MAX_DEPTH = args.max_depth
    SHOW = args.show
    MAX_EDGE = args.max_edge
    
    image = Image.open(image_path)
    depth_map = (np.array(image) / 255.0 * MAX_DEPTH).astype(np.float32)
    H, W = depth_map.shape
    mask = np.zeros((H, W), dtype=bool)
    final_depth = np.full_like(depth_map, np.nan)
    
    # 预生成网格坐标
    grid_v, grid_u = np.meshgrid(np.arange(W), np.arange(H))
    grid_v_flat = grid_v.ravel()
    grid_u_flat = grid_u.ravel()
    
    # 初始化累积点列表
    prev_points_list = []
    prev_depths_list = []
    
    for i in range(num_bins):
        low, high = depth_bins[i], depth_bins[i + 1]
        valid_mask = (depth_map >= low) & (depth_map < high) & (~mask)
        if not np.any(valid_mask):
            # print(f"跳过空区间: {low:.1f}-{high:.1f}m")
            continue
        
        u, v = np.where(valid_mask)
        depths = depth_map[valid_mask]
        current_points = np.column_stack((v.astype(np.float32), u.astype(np.float32)))
        current_depths = depths.astype(np.float32)
        
        # 合并之前的高区间点和当前点
        if len(prev_points_list) == 0:
            all_points = current_points
            all_depths = current_depths
        else:
            all_points = np.vstack([np.array(prev_points_list, dtype=np.float32), current_points])
            all_depths = np.hstack([np.array(prev_depths_list, dtype=np.float32), current_depths])
            
        if len(all_points) < 3:
            continue  # Delaunay需要至少3个点
        
        # Delaunay三角剖分
        try:
            tri = Delaunay(all_points, qhull_options="QJ QbB")  # 优化剖分参数
        except:
            continue
        
        # 向量化过滤三角形
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
            continue
        
        # 创建插值器
        tri_mpl = Triangulation(all_points[:, 0], all_points[:, 1], triangles=valid_tris)
        interpolator = LinearTriInterpolator(tri_mpl, all_depths)
        
        # 仅对未处理区域插值
        unmasked = ~mask
        interpolated = interpolator(grid_v[unmasked], grid_u[unmasked])
        valid_interp = ~np.isnan(interpolated)
        
        # 更新结果
        final_depth[unmasked] = np.where(valid_interp, interpolated, final_depth[unmasked])
        new_mask = unmasked.copy()
        new_mask[unmasked] = valid_interp
        mask |= new_mask
        
        # 更新prev_points为当前bin的高区间点
        prev_mask = (depth_map >= (high - 0.5)) & (depth_map < high) & (~mask)
        prev_u, prev_v = np.where(prev_mask)
        prev_depths = depth_map[prev_mask]
        prev_points_list = np.column_stack((prev_v, prev_u)).tolist()
        prev_depths_list = prev_depths.tolist()
    # 填充未处理区域为原始值
    final_depth = np.where(mask, final_depth, depth_map)
    plt_show_gray(final_depth, f"{file_name}-final", save_path, MAX_DEPTH)