# from func_utils.mesh_cpu import *
from func_utils.mesh_gpu_torch import *
from func_utils.depthdata import DepthDataset
# from func_utils.mesh_w_time import *
from func_utils.config import *

from tqdm import tqdm
from glob import glob
import os
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import DataLoader


if __name__ == '__main__':
    
    args = mesh_config()
    args.batch_size = 10
    dataset = DepthDataset(args.input_folder, args.max_depth)
    loader  = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=True)
    
    for batch in loader:
        depths, names = batch
        # 每张图单独处理
        for depth_map, fname in zip(depths, names):
            depth_map = depth_map.numpy()  # Tensor -> NumPy
            process_depth_map(args, depth_map,
                              args.output_folder, fname)

    process_folder(args)