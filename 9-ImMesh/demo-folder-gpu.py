import os
import time
import numpy as np
import torch
from scipy.spatial import Delaunay
from matplotlib.tri import Triangulation, LinearTriInterpolator
import cProfile, pstats
from numba import jit, prange
import multiprocessing as mp

# ======================================
# Step 1: Profiling Helper
# ======================================
def profile_pipeline(func, *args, **kwargs):
    """
    使用 cProfile 对 func(*args, **kwargs) 进行性能剖析，打印前 10 个最慢条目。
    返回 func 的执行结果。
    """
    pr = cProfile.Profile()
    pr.enable()
    result = func(*args, **kwargs)
    pr.disable()
    ps = pstats.Stats(pr).sort_stats('cumtime')
    ps.print_stats(10)
    return result

# 测试 Profiling
if __name__ == '__main__' and False:
    def dummy(n):
        s = 0
        for i in range(n): s += i**2
        return s
    profile_pipeline(dummy, 10_000_000)


# ======================================
# Step 2: GPU 加速的边长筛选
# ======================================
def filter_valid_tris_gpu(all_points: np.ndarray,
                           simplices: np.ndarray,
                           max_edge: float,
                           device: str='cuda') -> np.ndarray:
    """
    将三角网 all_points[simplices] 在 GPU 上并行算边长，筛选出最长边 <= max_edge 的三角。
    返回筛选后的 simplices。
    """
    pts = torch.from_numpy(all_points).float().to(device)      # [N_pts,2]
    simp = torch.from_numpy(simplices).long().to(device)      # [N_tri,3]
    tri_pts = pts[simp]                                      # [N_tri,3,2]

    # 计算三条边长
    e01 = (tri_pts[:,0] - tri_pts[:,1]).norm(dim=1)
    e12 = (tri_pts[:,1] - tri_pts[:,2]).norm(dim=1)
    e20 = (tri_pts[:,2] - tri_pts[:,0]).norm(dim=1)
    edges = torch.stack([e01, e12, e20], dim=1)             # [N_tri,3]

    # 筛选
    mask = edges.max(dim=1).values <= max_edge            # [N_tri]
    valid = simp[mask].cpu().numpy()                      # 回到 CPU 用于后续
    return valid

# 单元测试 GPU 筛选
def test_filter_valid_tris_gpu():
    pts = np.array([[0,0],[0,1],[1,0],[1,1]], float)
    simplices = np.array([[0,1,2],[1,2,3]], int)
    # 边长√2≈1.414, 都<=1.5
    out1 = filter_valid_tris_gpu(pts, simplices, max_edge=1.5, device='cuda')
    assert set(map(tuple,out1)) == {(0,1,2),(1,2,3)}
    # 要求<=1.0, 只有直角三角
    out2 = filter_valid_tris_gpu(pts, simplices, max_edge=1.0, device='cuda')
    assert set(map(tuple,out2)) == {(0,1,2)}
    print("test_filter_valid_tris_gpu passed.")


# ======================================
# Step 3: GPU 插值 Stub (可替换 CPU)
# ======================================
def interpolate_gpu_stub(all_points: np.ndarray,
                         all_depths: np.ndarray,
                         valid_tris: np.ndarray,
                         H: int, W: int,
                         mask: np.ndarray,
                         device: str='cuda') -> np.ndarray:
    """
    TODO: 在 GPU 上并行实现线性三角形插值。
    为保证功能，当前调用 CPU 版本。
    """
    return interpolate_cpu(all_points, all_depths, valid_tris, H, W, mask)

# 纯 CPU 插值实现（对照）
def interpolate_cpu(all_points: np.ndarray,
                    all_depths: np.ndarray,
                    valid_tris: np.ndarray,
                    H: int, W: int,
                    mask: np.ndarray) -> np.ndarray:
    """
    使用 matplotlib.tri.LinearTriInterpolator 对全图插值。
    对已有 mask 区域返回 nan。
    """
    tri = Triangulation(all_points[:,0], all_points[:,1], triangles=valid_tris)
    interp = LinearTriInterpolator(tri, all_depths)
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
    vals = interp(grid_x, grid_y)
    res = np.array(vals)
    res[mask] = np.nan
    return res

# CPU 插值单元测试
def test_interpolate_cpu():
    pts = np.array([[0,0],[0,1],[1,0]], float)
    deps = np.array([0,1,2], float)
    tris = np.array([[0,1,2]], int)
    H,W = 2,2
    mask = np.zeros((H,W), bool)
    out = interpolate_cpu(pts, deps, tris, H, W, mask)
    # (0,0)->0; (1,0)->deps at y=1,x=0 should be ~1
    assert np.allclose(out[0,0], 0)
    assert np.allclose(out[1,0], 1)
    print("test_interpolate_cpu passed.")


# ======================================
# Step 4: 获取未填充像素索引
# ======================================
def get_unfilled_indices(mask: np.ndarray) -> np.ndarray:
    """返回所有 mask==False 的像素坐标 (row,col) 列表。"""
    return np.stack(np.where(~mask), axis=1)

# 单元测试
def test_get_unfilled_indices():
    m = np.array([[True,False],[False,True]])
    idx = get_unfilled_indices(m)
    assert set(map(tuple, idx.tolist())) == {(0,1),(1,0)}
    print("test_get_unfilled_indices passed.")


# ======================================
# Step 5: GPU 上预分配示例（集成流程）
# ======================================
def process_one_image_accelerated(depth_map: np.ndarray,
                                  depth_bins: np.ndarray,
                                  max_edge: float,
                                  device: str='cuda') -> np.ndarray:
    """
    演示如何在 GPU 上预分配，并仅调用上面函数完成一个图像的插值。
    """
    H, W = depth_map.shape
    # 将数据搬到 GPU
    depth_t = torch.from_numpy(depth_map).float().to(device)
    final_t = torch.full((H, W), float('nan'), device=device)
    mask_t = torch.zeros((H, W), dtype=torch.bool, device=device)

    prev_pts = torch.empty((0,2), dtype=torch.float32, device=device)
    prev_deps = torch.empty((0,), dtype=torch.float32, device=device)

    for i in range(len(depth_bins)-1):
        low, high = depth_bins[i], depth_bins[i+1]
        valid_mask = (depth_t >= low) & (depth_t < high) & (~mask_t)
        if not valid_mask.any(): continue
        ys, xs = torch.where(valid_mask)
        pts = torch.stack((xs, ys), dim=1)
        deps = depth_t[ys, xs]

        all_pts = torch.cat((prev_pts, pts), dim=0).cpu().numpy()
        all_deps = torch.cat((prev_deps, deps), dim=0).cpu().numpy()

        # 更新 prev
        prev_zone = (depth_t >= (high-0.5)) & (depth_t < high) & (~mask_t)
        py, px = torch.where(prev_zone)
        prev_pts = torch.stack((px, py), dim=1)
        prev_deps = depth_t[py, px]

        # CPU Delaunay
        tri = Delaunay(all_pts)
        sims = tri.simplices
        valid_sims = filter_valid_tris_gpu(all_pts, sims, max_edge, device)

        # 插值
        cpu_mask = mask_t.cpu().numpy() if isinstance(mask_t, torch.Tensor) else mask_t
        interp = interpolate_gpu_stub(all_pts, all_deps, valid_sims, H, W, cpu_mask)
        interp_t = torch.from_numpy(interp).to(device)

        # 更新 final & mask
        new_region = (~torch.isnan(interp_t)) & (~mask_t)
        final_t[new_region] = interp_t[new_region]
        mask_t |= new_region

    # 填充剩余
    final_t[~mask_t] = depth_t[~mask_t]
    return final_t.cpu().numpy()

# 小规模功能测试
def test_process_one_image_accelerated():
    # 构造一个 4x4 人为深度图
    dm = np.array([
        [1,2,3,4],
        [2,3,4,5],
        [3,4,5,6],
        [4,5,6,7]
    ], float)
    bins = np.arange(1, 8, 2)
    out = process_one_image_accelerated(dm, bins, max_edge=1.5, device='cuda')
    assert out.shape == dm.shape
    print("test_process_one_image_accelerated passed.")


# ======================================
# Step 6: Numba 辅助的 CPU 版本边长筛选
# ======================================
@jit(nopython=True, parallel=True)
def filter_valid_tris_numba(all_pts: np.ndarray,
                            simplices: np.ndarray,
                            max_edge: float) -> np.ndarray:
    M = simplices.shape[0]
    out = np.empty((M,3), np.int64)
    cnt = 0
    for i in prange(M):
        a = all_pts[simplices[i,0]]
        b = all_pts[simplices[i,1]]
        c = all_pts[simplices[i,2]]
        d0 = np.sqrt(((a-b)**2).sum())
        d1 = np.sqrt(((b-c)**2).sum())
        d2 = np.sqrt(((c-a)**2).sum())
        if max(d0, d1, d2) <= max_edge:
            out[cnt,0] = simplices[i,0]
            out[cnt,1] = simplices[i,1]
            out[cnt,2] = simplices[i,2]
            cnt += 1
    return out[:cnt]

# 测试 Numba 版本
def test_filter_valid_tris_numba():
    pts = np.array([[0,0],[0,1],[1,0],[1,1]], float)
    sims = np.array([[0,1,2],[1,2,3]], int)
    out = filter_valid_tris_numba(pts, sims, 1.5)
    assert set(map(tuple,out.tolist())) == {(0,1,2),(1,2,3)}
    out2 = filter_valid_tris_numba(pts, sims, 1.0)
    assert set(map(tuple,out2.tolist())) == {(0,1,2)}
    print("test_filter_valid_tris_numba passed.")

# ======================================
# Step 7: 并行处理多张图片
# ======================================
def process_image_wrapper(args):
    img_path, bins, max_e = args
    # 读图略...
    # dm = ...
    # out = process_one_image_accelerated(dm, bins, max_e)
    # 保存结果...
    pass

if __name__ == '__main__':
    # 运行所有测试
    test_filter_valid_tris_gpu()
    test_interpolate_cpu()
    test_get_unfilled_indices()
    test_filter_valid_tris_numba()
    test_process_one_image_accelerated()
    print('All tests passed.')
