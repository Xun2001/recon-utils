# Concate bin with prev_points and prev_depths into all_points and all_depths

import os
import numpy as np
from PIL import Image
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from matplotlib.tri import LinearTriInterpolator, Triangulation
import argparse

def plt_show_gray(image_arr, title_name: str, save_path):
    """
    可视化灰度图像并保存为 PNG 文件。
    :param image_arr: 要可视化的图像数组
    :param title_name: 图像的标题，同时也是保存文件的名称
    :param save_path: 保存图像的路径
    """
    if save_path is not None:
        # 确保保存路径存在，如果不存在则创建
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 将数组中的 NaN、正无穷和负无穷值替换为 0
        image_arr = np.nan_to_num(image_arr, nan=0, posinf=0, neginf=0)
        # 将数组归一化到 0 - 255 范围并转换为 uint8 类型
        image_arr = (image_arr / MAX_DEPTH * 255).astype(np.uint8)
        # 将数组转换为 PIL 图像对象
        image = Image.fromarray(image_arr)
        # 构建保存图像的完整路径
        image_path = os.path.join(save_path, (title_name + '.png'))
        # 保存图像为 PNG 格式
        image.save(image_path, 'PNG')
        print("Saved png in", image_path)

# 解析命令行参数
parser = argparse.ArgumentParser(description='深度图插值脚本')
# 输入图像的路径，默认为 'sparse_depth.png'
parser.add_argument('--image_path', type=str, default='sparse_depth.png', help='输入图像的路径')
# 最大深度，默认为 30
parser.add_argument('--max_depth', type=float, default=30, help='最大深度')
# 最大允许三角形边长（像素），默认为 10
parser.add_argument('--max_edge', type=float, default=10, help='最大允许三角形边长（像素）')
# 是否保存中间结果
parser.add_argument('--show', action='store_true', help='是否保存中间结果')
# 分 BIN 的间距，默认为 1
parser.add_argument('--bin_interval', type=float, default=1, help='分 BIN 的间距')
# 保存路径，默认为指定的路径
parser.add_argument('--save_path', type=str, default='/home/qinllgroup/hongxiangyu/datasets/recon_utils/9-ImMesh/DEBUG/debug_MAX_10/save_bin_1', help='保存路径')
args = parser.parse_args()

# 参数配置
image_path = args.image_path
# 最大深度
MAX_DEPTH = args.max_depth
# 深度分界，根据分 BIN 的间距生成
depth_bins = np.arange(0.01, MAX_DEPTH + 0.51, args.bin_interval)
# 保存路径
save_path = args.save_path
# 是否保存中间结果
SHOW = args.show

# 初始化
# 打开输入图像
image = Image.open(image_path)
# 将图像转换为数组并归一化到最大深度范围，转换为 float32 类型
depth_map = (np.array(image) / 255.0 * MAX_DEPTH).astype(np.float32)
# 获取深度图的高度和宽度
H, W = depth_map.shape
# 计算分 BIN 的数量
num_bins = len(depth_bins) - 1
# 记录已处理区域的掩码，初始化为全 False
mask = np.zeros((H, W), dtype=bool)
# 最终深度图，初始化为全 NaN
final_depth = np.full_like(depth_map, np.nan)

# 用于存储上一个 BIN 的有效点和深度值
prev_points = np.array([]).reshape(0, 2)
prev_depths = np.array([])

# 按深度分层处理
for i in range(num_bins):
    # 获取当前 BIN 的深度范围
    low, high = depth_bins[i], depth_bins[i + 1]

    # 1. 筛选当前深度层且未被处理的原始点
    # 生成一个布尔掩码，标记当前深度层且未被处理的点
    valid_mask = (depth_map >= low) & (depth_map < high) & (~mask)
    if not np.any(valid_mask):
        # 如果当前深度层没有有效点，跳过该层
        print(f"跳过空区间: {low:.1f}-{high:.1f}m")
        continue

    # 2. 获取有效点坐标和深度值
    # 获取有效点的行和列索引
    u, v = np.where(valid_mask)
    # 获取有效点的深度值
    depths = depth_map[valid_mask]
    # 将行和列索引组合成 (x, y) 坐标
    points = np.column_stack((v, u))

    # 合并上一个 BIN 的有效点和深度值
    all_points = np.vstack((prev_points, points))
    all_depths = np.hstack((prev_depths, depths))

    if SHOW:
        # 如果需要保存中间结果，可视化并保存当前 BIN 的稀疏深度图
        pre_triangulation = np.full((H, W), np.nan, dtype=np.float32)
        pre_triangulation[u, v] = depths
        plt_show_gray(pre_triangulation, f'Bin-{i}-Sparse-Depth', save_path)

    # 3. 过滤离群点（使用三角形边长约束）
    try:
        # 对合并后的有效点进行 Delaunay 三角剖分
        tri = Delaunay(all_points)
    except:
        # 如果三角剖分失败，打印错误信息并跳过该层
        print("Delaunay 剖分失败")
        continue

    # 4. 过滤大三角形（基于边长阈值）
    # 最大允许边长（像素）
    MAX_EDGE = args.max_edge
    # 存储有效三角形的索引
    valid_tris = []
    for simplex in tri.simplices:
        # 获取三角形的三个顶点坐标
        a, b, c = all_points[simplex]
        # 计算三角形三条边的长度
        edges = [
            np.linalg.norm(a - b),
            np.linalg.norm(b - c),
            np.linalg.norm(c - a)
        ]
        if max(edges) <= MAX_EDGE:
            # 如果三角形的最大边长小于等于阈值，将其添加到有效三角形列表中
            valid_tris.append(simplex)

    if len(valid_tris) == 0:
        # 如果没有有效三角形，打印错误信息并跳过该层
        print("无有效三角形")
        continue

    # 5. 创建过滤后的三角剖分
    # 使用合并后的有效点和有效三角形创建 Matplotlib 的三角剖分对象
    tri_mpl = Triangulation(all_points[:, 0], all_points[:, 1], triangles=valid_tris)
    # 创建线性三角插值器
    interpolator = LinearTriInterpolator(tri_mpl, all_depths)

    # 6. 网格插值
    # 生成网格坐标
    grid_v, grid_u = np.meshgrid(np.arange(W), np.arange(H))
    # 对网格坐标进行插值
    interpolated = interpolator(grid_v.ravel(), grid_u.ravel())
    # 将插值结果重塑为与深度图相同的形状
    interp_image = interpolated.reshape((H, W))

    # 7. 更新处理区域：只保留新增的有效区域
    # 生成一个布尔掩码，标记新增的有效区域
    new_region = (~np.isnan(interp_image)) & (~mask)
    # 将新增有效区域的插值结果更新到最终深度图中
    final_depth[new_region] = interp_image[new_region]
    # 更新全局掩码
    mask |= new_region

    if SHOW:
        # 如果需要保存中间结果，可视化并保存当前 BIN 的插值结果
        plt_show_gray(interp_image, f'Bin-{i}-Inter', save_path)

    # 更新上一个 BIN 的有效点和深度值
    prev_points = points
    prev_depths = depths

# 填充未被处理的区域
# 将未被处理的区域用原始深度图的值填充
final_depth = np.where(mask, final_depth, depth_map)

# 最终结果处理
# 可视化并保存最终深度图
plt_show_gray(final_depth, 'Final_depth', save_path)