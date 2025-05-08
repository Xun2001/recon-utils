import os
import time
import json
import argparse
import numpy as np
from PIL import Image
from scipy.spatial import Delaunay
from matplotlib.tri import Triangulation, LinearTriInterpolator
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm

from process_utils.config import mesh_config

def save_gray(arr: np.ndarray, name: str, save_folder: str, max_depth: float):
    """把深度图归一化到 [0,255] 并保存为 PNG。"""
    img = np.nan_to_num(arr, nan=0.0)
    img = (img / max_depth * 255.0).clip(0, 255).astype(np.uint8)
    im = Image.fromarray(img)
    im.save(os.path.join(save_folder, f"{name}.png"))

def process_bin(points: np.ndarray,
                depths: np.ndarray,
                grid_x: np.ndarray,
                grid_y: np.ndarray,
                max_edge: float) -> tuple:
    """
    处理单个 bin 并返回插值结果和时间统计
    """
    time_stats = {}
    
    # Delaunay 三角剖分
    start_time = time.time()
    tri = Delaunay(points)
    simplices = tri.simplices  # (M,3)
    time_stats['delaunay'] = time.time() - start_time
    
    # 边长过滤
    start_time = time.time()
    pts = points[simplices]  # (M,3,2)
    edges = pts - pts[:, [1,2,0], :]  # 三个边向量
    lengths = np.linalg.norm(edges, axis=2)  # (M,3)
    max_len = lengths.max(axis=1)           # (M,)
    keep = max_len <= max_edge
    valid_simplices = simplices[keep]
    time_stats['filter'] = time.time() - start_time
    
    # 插值
    start_time = time.time()
    tri2d = Triangulation(points[:,0], points[:,1], triangles=valid_simplices)
    interp = LinearTriInterpolator(tri2d, depths)
    xi = grid_x.ravel()
    yi = grid_y.ravel()
    zi = interp(xi, yi)               # (H*W,)
    time_stats['interp'] = time.time() - start_time
    
    # 总时间（用于校验）
    time_stats['total'] = sum(time_stats.values())
    
    return zi.reshape(grid_x.shape), time_stats

def parallel_depth_fill(bins_data: list,
                        grid_x: np.ndarray,
                        grid_y: np.ndarray,
                        max_edge: float,
                        num_workers: int) -> tuple:
    """
    返回插值结果和时间统计字典
    """
    futures = {}
    with ProcessPoolExecutor(max_workers=num_workers) as exe:
        for bin_data in bins_data:
            fut = exe.submit(
                process_bin,
                bin_data['points'], 
                bin_data['depths'],
                grid_x, 
                grid_y,
                max_edge
            )
            futures[fut] = bin_data['depth_value']

        results = []
        all_time_stats = []
        for fut in as_completed(futures):
            dval = futures[fut]
            zi, time_stats = fut.result()
            results.append((dval, zi))
            all_time_stats.append(time_stats)

    # 结果排序（近→远）
    results.sort(key=lambda x: x[0], reverse=False)

    H, W = grid_x.shape
    final = np.full((H, W), np.nan, dtype=float)
    mask  = np.zeros((H, W), dtype=bool)

    for _, zi in results:
        region = ~np.isnan(zi)
        new_region = region & (~mask)
        final[new_region] = zi[new_region]
        mask |= new_region

    # 时间统计汇总
    total_delaunay = sum(ts['delaunay'] for ts in all_time_stats)
    total_filter = sum(ts['filter'] for ts in all_time_stats)
    total_interp = sum(ts['interp'] for ts in all_time_stats)
    total_bins_time = total_delaunay + total_filter + total_interp
    
    time_info = {
        'total_delaunay': total_delaunay,
        'total_filter': total_filter,
        'total_interp': total_interp,
        'total_bins_time': total_bins_time,
        'num_bins_processed': len(all_time_stats),
        'avg_per_bin': {
            'delaunay': total_delaunay / len(all_time_stats),
            'filter': total_filter / len(all_time_stats),
            'interp': total_interp / len(all_time_stats),
            'total': total_bins_time / len(all_time_stats)
        }
    }

    return final, time_info

def process_image(png_path: str,
                  num_bins: int,
                  max_depth: float,
                  max_edge: float,
                  bin_workers: int,
                  output_folder: str,
                  show: bool) -> None:
    """
    添加分bin构造时间统计和填充时间统计
    """
    fname = os.path.splitext(os.path.basename(png_path))[0]
    t0 = time.time()

    # 1) 读取图片
    start_read = time.time()
    image = Image.open(png_path)
    depth_map = (np.array(image) / 255.0 * max_depth).astype(np.float32)
    H, W = depth_map.shape
    t_read = time.time() - start_read

    # 2) 分bin构造
    start_bin_construction = time.time()
    nonzero = depth_map > 0
    if not nonzero.any():
        print(f"{fname}: 全零图，跳过")
        return
    
    dmin = depth_map[nonzero].min()
    dmax = depth_map.max()
    edges = np.linspace(dmin, dmax, num_bins + 1)
    
    ys, xs = np.nonzero(depth_map >= dmin)
    pts = np.column_stack((xs, ys))
    ds  = depth_map[ys, xs]
    
    overlap = 0.2
    bins_data = []
    for i in range(num_bins):
        low = edges[i]
        high = edges[i+1]
        if i>0:
            low = low - overlap
        low = max(low, dmin)
        sel = np.flatnonzero((ds >= low) & (ds < high))
        if sel.size == 0:
            continue
        bins_data.append({
            'points': pts[sel],
            'depths': ds[sel],
            'depth_value': low
        })
    t_bin_construction = time.time() - start_bin_construction

    # 3) 并行处理
    start_parallel = time.time()
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
    interpolated, bins_time_info = parallel_depth_fill(
        bins_data, grid_x, grid_y,
        max_edge, bin_workers
    )
    t_parallel = time.time() - start_parallel

    # 4) 填充剩余区域
    start_fill = time.time()
    final = np.where(np.isnan(interpolated), depth_map, interpolated)
    t_fill = time.time() - start_fill

    # 保存结果
    save_gray(final, f"{fname}-final", output_folder, max_depth)

    # 时间统计
    t_total = time.time() - t0
    stats = {
        'image_reading': t_read,
        'bin_construction': t_bin_construction,
        'parallel_processing': {
            'total_time': t_parallel,
            'delaunay_total': bins_time_info['total_delaunay'],
            'filter_total': bins_time_info['total_filter'],
            'interp_total': bins_time_info['total_interp'],
            'sum_subprocess_time': bins_time_info['total_bins_time'],
            'parallel_overhead': t_parallel - bins_time_info['total_bins_time'],
            'avg_per_bin': bins_time_info['avg_per_bin'],
            'num_bins_processed': bins_time_info['num_bins_processed']
        },
        'filling_time': t_fill,
        'total_processing': t_total
    }

    with open(os.path.join(output_folder,"json_files", f"{fname}_time_stats.json"), 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"{fname} → 完成（共 {t_total:.1f}s）")


def main():

    args = mesh_config()

    save_path = os.path.join(args.output_folder,f"{args.process_fun}_{int(args.num_image_workers)}_{int(args.bin_workers)}")
    os.makedirs(save_path, exist_ok=True)
    
    print(f"输出保存的文件夹: {save_path}")
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path,"json_files"), exist_ok=True)
    
    # 最外层并行处理多张图
    pngs = sorted(
    os.path.join(args.input_folder, f)
    for f in os.listdir(args.input_folder)
    if f.lower().endswith('.png')
    )

    folder_start_time = time.time()
    with ThreadPoolExecutor(max_workers=args.num_image_workers) as exe:
        futures = {
            exe.submit(process_image,
                       png_path,
                       args.num_bins,
                       args.max_depth,
                       args.max_edge,
                       args.bin_workers,
                       save_path,
                       args.show): png_path
            for png_path in pngs
        }
        for fut in as_completed(futures):
            _ = fut.result()
    
    print('Done.')
    total_time = time.time() - folder_start_time
    print(f"Time taken: {total_time:.2f}s")
    
    time_data = {
    "total_time": total_time
    }
    with open(os.path.join(save_path, f"{args.num_image_workers}-{args.bin_workers}_time.json"), 'w') as f:
        json.dump(time_data, f, indent=2)


# main函数保持不变
if __name__ == "__main__":
    main()