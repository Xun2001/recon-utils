import os
import numpy as np
from PIL import Image
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from matplotlib.tri import LinearTriInterpolator, Triangulation

def plt_show_gray(image_arr, title_name: str, save_path):
    """可视化灰度图像并保存"""
    plt.figure()
    plt.imshow(image_arr, cmap='gray')
    plt.title(title_name)
    plt.colorbar()
    plt.show()
    
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_arr = np.nan_to_num(image_arr, nan=0, posinf=0, neginf=0)
        image_arr = (image_arr / MAX_DEPTH * 255).astype(np.uint8)
        image = Image.fromarray(image_arr)
        image_path = os.path.join(save_path, (title_name + '.png'))
        image.save(image_path, 'PNG')
        print("Saved png in", image_path)

# 参数配置
image_path = 'sparse_depth.png'
MAX_DEPTH = 30
depth_bins = np.arange(0.01, MAX_DEPTH + 0.51, 5)  # 深度分界
save_path = '/home/qinllgroup/hongxiangyu/datasets/recon_utils/9-ImMesh/debug_save'
DEBUG = True
SHOW = True

# 初始化
image = Image.open(image_path)
depth_map = (np.array(image) / 255.0 * MAX_DEPTH).astype(np.float32)
H, W = depth_map.shape
num_bins = len(depth_bins) - 1
mask = np.zeros((H, W), dtype=bool)  # 记录已处理区域
final_depth = np.full_like(depth_map, np.nan)  # 最终深度图

# 按深度分层处理
for i in range(num_bins):
    low, high = depth_bins[i], depth_bins[i+1]
    
    # 1. 筛选当前深度层且未被处理的原始点
    valid_mask = (depth_map >= low) & (depth_map < high) & (~mask)
    if not np.any(valid_mask):
        print(f"跳过空区间: {low:.1f}-{high:.1f}m")
        continue
    
    # 2. 获取有效点坐标和深度值
    u, v = np.where(valid_mask)
    depths = depth_map[valid_mask]
    points = np.column_stack((v, u))  # 转换为(x,y)坐标
    
    # 3. 过滤离群点（使用三角形边长约束）
    if len(points) < 3:
        print(f"点数不足({len(points)})无法三角化")
        continue
    
    try:
        tri = Delaunay(points)
    except:
        print("Delaunay剖分失败")
        continue
    
    # 4. 过滤大三角形（基于边长阈值）
    MAX_EDGE = 10  # 最大允许边长（像素）
    valid_tris = []
    for simplex in tri.simplices:
        a, b, c = points[simplex]
        edges = [
            np.linalg.norm(a-b),
            np.linalg.norm(b-c),
            np.linalg.norm(c-a)
        ]
        if max(edges) <= MAX_EDGE:
            valid_tris.append(simplex)
    
    if len(valid_tris) == 0:
        print("无有效三角形")
        continue
    
    # 5. 创建过滤后的三角剖分
    tri_mpl = Triangulation(points[:,0], points[:,1], triangles=valid_tris)
    interpolator = LinearTriInterpolator(tri_mpl, depths)
    
    # 6. 网格插值
    grid_v, grid_u = np.meshgrid(np.arange(W), np.arange(H))
    interpolated = interpolator(grid_v.ravel(), grid_u.ravel())
    interp_image = interpolated.reshape((H, W))
    
    # 7. 更新处理区域：只保留新增的有效区域
    new_region = (~np.isnan(interp_image)) & (~mask)
    final_depth[new_region] = interp_image[new_region]
    mask |= new_region  # 更新全局掩码
    
    if SHOW:
        plt_show_gray(interp_image, f'Bin {i}插值结果', save_path)

# 填充未被处理的区域
final_depth = np.where(mask, final_depth, depth_map)

# 最终结果处理
plt_show_gray(final_depth, '最终深度图', save_path)