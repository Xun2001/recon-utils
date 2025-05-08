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
                max_edge: float) -> np.ndarray:
    """
    单个 bin 的处理：Delaunay → 边长过滤 → Triangulation + 插值 → 返回 HxW 网格上的深度值（NaN 表示无数据）
    """
    # 1) Delaunay
    tri = Delaunay(points)
    simplices = tri.simplices  # (M,3)

    # 2) 边长过滤
    pts = points[simplices]  # (M,3,2)
    edges = pts - pts[:, [1,2,0], :]  # 三个边向量
    lengths = np.linalg.norm(edges, axis=2)  # (M,3)
    max_len = lengths.max(axis=1)           # (M,)
    keep = max_len <= max_edge
    valid_simplices = simplices[keep]

    # 3) 插值
    tri2d = Triangulation(points[:,0], points[:,1], triangles=valid_simplices)
    interp = LinearTriInterpolator(tri2d, depths)
    xi = grid_x.ravel()
    yi = grid_y.ravel()
    zi = interp(xi, yi)               # (H*W,)
    return zi.reshape(grid_x.shape)   # (H,W) zi.reshape(H,W)

def parallel_depth_fill(bins_data: list,
                        grid_x: np.ndarray,
                        grid_y: np.ndarray,
                        max_edge: float,
                        num_workers: int) -> np.ndarray:
    """
    并行地对每个 bin 调用 process_bin，收集所有 (depth_value, zi) 结果后，
    按 depth_value 从大到小（最远→最近）排序，再逐层叠加到 final 中。
    """
    futures = {}
    with ProcessPoolExecutor(max_workers=num_workers) as exe:
        for bin_data in bins_data:
            fut = exe.submit(process_bin,
                             bin_data['points'], bin_data['depths'],
                             grid_x, grid_y,
                             max_edge)
            futures[fut] = bin_data['depth_value']

        results = []
        for fut in as_completed(futures):
            dval = futures[fut]
            zi = fut.result()
            results.append((dval, zi))

    # 从最远到最近排序，确保“近”覆盖“远”
    # results.sort(key=lambda x: x[0], reverse=True) # need-check
    results.sort(key=lambda x: x[0], reverse=False)

    H, W = grid_x.shape
    final = np.full((H, W), np.nan, dtype=float)
    mask  = np.zeros((H, W), dtype=bool)

    for _, zi in results:
        region     = ~np.isnan(zi)
        new_region = region & (~mask)
        final[new_region] = zi[new_region]
        mask |= new_region

    return final

def process_image(png_path: str,
                  num_bins: int,
                  max_depth: float,
                  max_edge: float,
                  bin_workers: int,
                  output_folder: str,
                  show: bool) -> None:
    """
    单张图片处理：
      1. 读灰度 PNG → depth_map
      2. 按 depth_bins 分箱，构造 bins_data
      3. 调用 parallel_depth_fill 得到分层插值结果
      4. 用原 depth_map 填充剩余未插值区域
      5. 保存最终结果 & 可选中间图 & 时间统计
    """
    fname = os.path.splitext(os.path.basename(png_path))[0]
    t0 = time.time()

    # 1) 读取 & 转深度
    image = Image.open(png_path)
    depth_map = (np.array(image) / 255.0 * max_depth).astype(np.float32)
    H, W = depth_map.shape

    t_read = time.time() - t0

    # 2) 分 bin
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
    # idxs = np.digitize(ds, edges) - 1  # bin index 0...num_bins-1

    overlap = 0.2
    bins_data = []
    for i in range(num_bins):
        low = edges[i]
        high = edges[i+1]
        if i>0:
            low = low - overlap
        low = max(low,dmin)
        sel = np.flatnonzero((ds >= low) & (ds < high))# selected points
        if sel.size == 0:
            continue
        depth_value = low # key for sorting
        bins_data.append({
            'points':      pts[sel],
            'depths':      ds[sel],
            'depth_value': depth_value       # 用 bin 下界做排序 key
        })

    # 可视化每个 bin 原始稀疏点（可选）
    if show:
        for i, bd in enumerate(bins_data):
            tmp = np.full((H, W), np.nan, dtype=np.float32)
            xs_i, ys_i = bd['points'][:,0], bd['points'][:,1]
            tmp[ys_i, xs_i] = bd['depths']
            save_gray(tmp, f"{fname}-bin{i}-sparse", output_folder, max_depth)

    # 3) 并行剖分 + 插值
    t_bin0 = time.time()
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
    interpolated = parallel_depth_fill(
        bins_data, grid_x, grid_y,
        max_edge, bin_workers
    )
    t_bins = time.time() - t_bin0

    # 4) 填充剩余区域
    final = np.where(np.isnan(interpolated), depth_map, interpolated)

    # 5) 保存 & 统计
    save_gray(final, f"{fname}-final", output_folder, max_depth)
    t_total = time.time() - t0

    stats = {
        'image_reading':    t_read,
        'bins_processing':  t_bins,
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

if __name__ == '__main__':
    main()
