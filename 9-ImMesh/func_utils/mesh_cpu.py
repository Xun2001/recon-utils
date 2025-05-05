import os
import numpy as np
from PIL import Image
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from matplotlib.tri import LinearTriInterpolator, Triangulation
import argparse
from tqdm import tqdm


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
        # print("Saved png in", image_path)

def process_image_bin_points(args, depth_bins,num_bins, depth_image_path, save_path, file_name):
    MAX_DEPTH = args.max_depth
    SHOW = args.show
    
    depth_image = Image.open(depth_image_path)
    depth_map = (np.array(depth_image) / 255.0 * MAX_DEPTH).astype(np.float32)
    H, W = depth_map.shape
    mask = np.zeros((H, W), dtype=bool)
    final_depth = np.full_like(depth_map, np.nan)
    
    for i in range(num_bins):
        low, high = depth_bins[i], depth_bins[i + 1]
        valid_mask = (depth_map >= low) & (depth_map < high) & (~mask)
        if not np.any(valid_mask):
            # print(f"跳过空区间: {low:.1f}-{high:.1f}m")
            continue
        u, v = np.where(valid_mask)
        depths = depth_map[valid_mask]
        points = np.column_stack((v, u))
        
        if SHOW:
            pre_triangulation = np.full((H, W), np.nan, dtype=np.float32)
            pre_triangulation[u, v] = depths
            plt_show_gray(pre_triangulation, f'Bin-{i}-Sparse-Depth', save_path, MAX_DEPTH)
        try:
            tri = Delaunay(points)
        except:
            print("Delaunay 剖分失败")
            continue
        MAX_EDGE = args.max_edge
        valid_tris = []
        for simplex in tri.simplices:
            a, b, c = points[simplex]
            edges = [
                np.linalg.norm(a - b),
                np.linalg.norm(b - c),
                np.linalg.norm(c - a)
            ]
            if max(edges) <= MAX_EDGE:
                valid_tris.append(simplex)
                
        if len(valid_tris) == 0:
            # print("无有效三角形")
            continue
        tri_mpl = Triangulation(points[:, 0], points[:, 1], triangles=valid_tris)
        interpolator = LinearTriInterpolator(tri_mpl, depths)
        grid_v, grid_u = np.meshgrid(np.arange(W), np.arange(H))
        interpolated = interpolator(grid_v.ravel(), grid_u.ravel())
        interp_image = interpolated.reshape((H, W))
        new_region = (~np.isnan(interp_image)) & (~mask)
        final_depth[new_region] = interp_image[new_region]
        mask |= new_region
        
        if SHOW:
            plt_show_gray(interp_image, f'Bin-{i}-Inter', save_path, MAX_DEPTH)
    final_depth = np.where(mask, final_depth, depth_map)
    plt_show_gray(final_depth, f"{file_name}-final", save_path, MAX_DEPTH)

def process_image_bin2_points(args, depth_bins,num_bins, depth_image_path, save_path, file_name):
    """
    相较于process_image_bin_points
    会对Bin进行扩展
    Args:
        args (_type_): _description_
        depth_bins (_type_): _description_
        num_bins (_type_): _description_
        depth_image_path (_type_): _description_
        save_path (_type_): _description_
        file_name (_type_): _description_
    """
    MAX_DEPTH = args.max_depth
    SHOW = args.show
    
    depth_image = Image.open(depth_image_path)
    depth_map = (np.array(depth_image) / 255.0 * MAX_DEPTH).astype(np.float32)
    H, W = depth_map.shape
    mask = np.zeros((H, W), dtype=bool)
    final_depth = np.full_like(depth_map, np.nan)
    
    for i in range(num_bins):
        
        low, high = depth_bins[i], depth_bins[i + 1]
        # 扩展 bin 范围
        ## 当前 bin
        current_mask = (depth_map >= low) & (depth_map < high) & (~mask)
        u_curr, v_curr = np.where(current_mask)
        depths_curr = depth_map[current_mask]
        points_curr = np.column_stack((v_curr, u_curr))
        ## 前 bin 0.5
        prev_low = max(low - 0.5, 0.0)  # 防止负数
        prev_mask = (depth_map >= prev_low) & (depth_map < low) & (~mask)
        u_prev, v_prev = np.where(prev_mask)
        depths_prev = depth_map[prev_mask]
        points_prev = np.column_stack((v_prev, u_prev))
        ## 3. 合并点集
        all_points = np.vstack((points_prev, points_curr))
        all_depths = np.hstack((depths_prev, depths_curr))
        
        if len(all_points) == 0:
            # print(f"跳过空区间: {low:.1f}-{high:.1f}m (无有效点)")
            continue
        
        if SHOW:
            pre_triangulation = np.full((H, W), np.nan, dtype=np.float32)
            pre_triangulation[u_prev, v_prev] = all_depths
            plt_show_gray(pre_triangulation, f'Bin-{i}-Sparse-Depth', save_path, MAX_DEPTH)
        try:
            tri = Delaunay(all_points)
        except:
            print(f"BIN {i} with {len(all_points)} points")
            print("Delaunay 剖分失败")
            continue
        MAX_EDGE = args.max_edge
        valid_tris = []
        for simplex in tri.simplices:
            a, b, c = all_points[simplex]
            edges = [
                np.linalg.norm(a - b),
                np.linalg.norm(b - c),
                np.linalg.norm(c - a)
            ]
            if max(edges) <= MAX_EDGE:
                valid_tris.append(simplex)
                
        if len(valid_tris) == 0:
            # print("无有效三角形")
            continue
        tri_mpl = Triangulation(all_points[:, 0], all_points[:, 1], triangles=valid_tris)
        interpolator = LinearTriInterpolator(tri_mpl, all_depths)
        grid_v, grid_u = np.meshgrid(np.arange(W), np.arange(H))
        interpolated = interpolator(grid_v.ravel(), grid_u.ravel())
        interp_image = interpolated.reshape((H, W))
        
        new_region = current_mask.copy()
        # new_region = (~np.isnan(interp_image)) & (~mask)
        final_depth[new_region] = interp_image[new_region]
        mask |= new_region
        
        if SHOW:
            plt_show_gray(interp_image, f'Bin-{i}-Inter', save_path, MAX_DEPTH)
    final_depth = np.where(mask, final_depth, depth_map)
    plt_show_gray(final_depth, f"{file_name}-final", save_path, MAX_DEPTH)


def process_image_stack_points(args,depth_bins,num_bins, image_path, save_path, file_name):
    """
    处理一张深度图，使用 Delaunay 剖分和三角插值进行深度插值。
    会额外增加前一个bin的点

    Args:
        image_path (_type_): _description_
        save_path (_type_): _description_
        file_name (_type_): _description_
    """
    MAX_DEPTH = args.max_depth
    SHOW = args.show
    
    image = Image.open(image_path)
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

        if SHOW:
            pre_triangulation = np.full((H, W), np.nan, dtype=np.float32)
            pre_triangulation[u, v] = depths
            plt_show_gray(pre_triangulation, f'Bin-{i}-Sparse-Depth', save_path, MAX_DEPTH)

        try:
            tri = Delaunay(all_points)
        except:
            print("Delaunay 剖分失败")
            continue

        MAX_EDGE = args.max_edge
        valid_tris = []
        for simplex in tri.simplices:
            a, b, c = all_points[simplex]
            edges = [
                np.linalg.norm(a - b),
                np.linalg.norm(b - c),
                np.linalg.norm(c - a)
            ]
            if max(edges) <= MAX_EDGE:
                valid_tris.append(simplex)

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
            plt_show_gray(interp_image, f'Bin-{i}-Inter', save_path, MAX_DEPTH)

        prev_points = points
        prev_depths = depths

    final_depth = np.where(mask, final_depth, depth_map)
    plt_show_gray(final_depth, f"{file_name}-final", save_path, MAX_DEPTH)

def process_image_stack2_points(args,depth_bins,num_bins, image_path, save_path, file_name):
    """
    处理一张深度图，使用 Delaunay 剖分和三角插值进行深度插值。
    只增加前一个 bin 中 0.5m 距离内的点

    Args:
        image_path (_type_): _description_
        save_path (_type_): _description_
        file_name (_type_): _description_
    """
    MAX_DEPTH = args.max_depth
    SHOW = args.show
    
    image = Image.open(image_path)
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
        except:
            print("Delaunay 剖分失败")
            continue

        MAX_EDGE = args.max_edge
        valid_tris = []
        for simplex in tri.simplices:
            a, b, c = all_points[simplex]
            edges = [
                np.linalg.norm(a - b),
                np.linalg.norm(b - c),
                np.linalg.norm(c - a)
            ]
            if max(edges) <= MAX_EDGE:
                valid_tris.append(simplex)

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
            plt_show_gray(interp_image, f'Bin-{i}-Inter', save_path, MAX_DEPTH)


    final_depth = np.where(mask, final_depth, depth_map)
    plt_show_gray(final_depth, f"{file_name}-final", save_path, MAX_DEPTH)


def process_image_stack_acc_points(args,depth_bins,num_bins, image_path, save_path, file_name):
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
    
    image = Image.open(image_path)
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
            plt_show_gray(interp_image, f'Bin-{i}-Inter', save_path, MAX_DEPTH)


    final_depth = np.where(mask, final_depth, depth_map)
    plt_show_gray(final_depth, f"{file_name}-final", save_path, MAX_DEPTH)