import os
import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from scipy.spatial import Delaunay
from matplotlib.tri import LinearTriInterpolator, Triangulation
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 配置参数
class Args:
    max_depth = 30.0          # 最大深度值
    max_edge = 10.0           # 三角形最大边长阈值
    num_bins = 30             # 深度区间数量
    show = False              # 不显示中间结果
    num_workers = 4           # 并行工作进程数
    batch_size = 4            # 批处理大小

# 自定义数据集实现并行读取
class DepthDataset(Dataset):
    def __init__(self, image_folder):
        self.image_paths = [
            os.path.join(image_folder, f) 
            for f in os.listdir(image_folder) 
            if f.endswith('.png')
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None, None
        return image, os.path.basename(path)

# GPU加速的插值核函数
@torch.jit.script
def gpu_interpolate(grid_points, triangles, depths, H: int, W: int):
    device = triangles.device
    final_depth = torch.full((H*W,), torch.nan, device=device)
    
    # 计算三角形包围盒
    min_xy = torch.min(triangles, dim=1)[0]
    max_xy = torch.max(triangles, dim=1)[0]
    
    for i in range(triangles.shape[0]):
        # 获取当前三角形的顶点和深度
        tri = triangles[i]
        z = depths[i]
        
        # 计算包围盒
        x_min, y_min = min_xy[i]
        x_max, y_max = max_xy[i]
        
        # 获取包围盒内的所有像素
        in_box = (
            (grid_points[:,0] >= x_min) & 
            (grid_points[:,0] <= x_max) & 
            (grid_points[:,1] >= y_min) & 
            (grid_points[:,1] <= y_max)
        )
        candidates = grid_points[in_box]
        
        # 计算重心坐标
        v0 = tri[2] - tri[0]
        v1 = tri[1] - tri[0]
        v2 = candidates - tri[0]
        
        dot00 = v0 @ v0.T
        dot01 = v0 @ v1.T
        dot02 = v0 @ v2.T
        dot11 = v1 @ v1.T
        dot12 = v1 @ v2.T
        
        invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * invDenom
        v = (dot00 * dot12 - dot01 * dot02) * invDenom
        
        # 判断是否在三角形内
        mask = (u >= 0) & (v >= 0) & (u + v <= 1)
        valid_indices = in_box.nonzero()[0][mask]
        
        # 插值计算深度值
        w = 1 - u[mask] - v[mask]
        interp_z = w*z[0] + u[mask]*z[1] + v[mask]*z[2]
        
        # 更新深度图（保留最先插值的）
        final_depth[valid_indices] = torch.where(
            torch.isnan(final_depth[valid_indices]),
            interp_z,
            final_depth[valid_indices]
        )
    
    return final_depth.reshape(H, W)

def process_batch(args, images, names, save_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    depth_bins = torch.linspace(0, args.max_depth, args.num_bins+1).to(device)
    
    for idx, (image, name) in enumerate(zip(images, names)):
        if image is None:
            continue
            
        # 转换为PyTorch张量并传输到GPU
        depth_map = (torch.from_numpy(image.astype(np.float32)) / 255 * args.max_depth).to(device)
        H, W = depth_map.shape
        grid_v, grid_u = torch.meshgrid(torch.arange(W), torch.arange(H))
        grid_points = torch.stack([grid_v.flatten(), grid_u.flatten()], dim=1).float().to(device)
        
        mask = torch.zeros_like(depth_map, dtype=torch.bool)
        final_depth = torch.full_like(depth_map, torch.nan)
        prev_points = torch.empty((0,2), device=device)
        prev_depths = torch.empty((0,), device=device)
        
        for i in range(args.num_bins):
            low, high = depth_bins[i], depth_bins[i+1]
            
            # 创建有效掩膜
            valid_mask = (depth_map >= low) & (depth_map < high) & ~mask
            if not torch.any(valid_mask):
                continue
                
            # 获取有效点
            u, v = torch.where(valid_mask)
            points = torch.stack([v, u], dim=1).float()
            depths = depth_map[valid_mask]
            
            # 合并历史点
            all_points = torch.cat([prev_points, points], dim=0)
            all_depths = torch.cat([prev_depths, depths], dim=0)
            
            # 更新历史点为当前区间上部0.5m的点
            prev_mask = (depth_map >= (high-0.5)) & (depth_map < high) & ~mask
            prev_u, prev_v = torch.where(prev_mask)
            prev_points = torch.stack([prev_v, prev_u], dim=1).float()
            prev_depths = depth_map[prev_mask]
            
            # CPU上进行Delaunay三角剖分
            all_points_cpu = all_points.cpu().numpy()
            try:
                tri = Delaunay(all_points_cpu)
            except:
                continue
                
            simplices = torch.from_numpy(tri.simplices).to(device)
            
            # 过滤过大三角形
            pts = all_points[simplices]
            edges = pts - pts[:, [1,2,0], :]
            edge_lengths = torch.norm(edges, dim=2)
            max_lengths, _ = torch.max(edge_lengths, dim=1)
            valid_tris = simplices[max_lengths <= args.max_edge]
            
            if valid_tris.size(0) == 0:
                continue
                
            # 准备插值数据
            tri_points = all_points[valid_tris]
            tri_depths = all_depths[valid_tris]
            
            # GPU加速插值
            interp_depth = gpu_interpolate(
                grid_points, tri_points, tri_depths, H, W
            )
            
            # 更新结果
            new_region = ~torch.isnan(interp_depth) & ~mask
            final_depth[new_region] = interp_depth[new_region]
            mask |= new_region
            
        # 合并结果并保存
        final_depth = torch.where(mask, final_depth, depth_map)
        output_path = os.path.join(save_path, f"{name.split('.')[0]}-final.png")
        cv2.imwrite(output_path, (final_depth.cpu().numpy() / args.max_depth * 255).astype(np.uint8))

def main():
    args = Args()
    dataset = DepthDataset("your_image_folder_path")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=lambda x: [(img, name) for img, name in x if img is not None]
    )
    
    os.makedirs("output_folder", exist_ok=True)
    
    # 使用多GPU加速
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU进行加速")
        mp.set_start_method('spawn', force=True)
        
    for batch in tqdm(dataloader):
        images, names = zip(*batch)
        process_batch(args, images, names, "output_folder")

if __name__ == "__main__":
    main()