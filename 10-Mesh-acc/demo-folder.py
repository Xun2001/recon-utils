from process_utils.config import mesh_config
from process_utils.process_depth import *

import os
import tqdm
from torch.utils.data import Dataset,DataLoader
from concurrent.futures import ThreadPoolExecutor


class DepthProcessingDataset(Dataset):
    def __init__(self, file_list, input_folder, output_folder, args, depth_bins, num_bins):
        self.file_list = file_list
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.args = args
        self.depth_bins = depth_bins
        self.num_bins = num_bins

    def __len__(self):
        return len(self.file_list)



if __name__ == '__main__':
    
    args = mesh_config()
    args.image_folder = os.path.join(args.input_folder, 'images')
    args.depth_folder = os.path.join(args.input_folder, 'depths')
    args.use_gpu = True
    
    
    depth_folder = os.path.join(args.input_folder, 'depths')
    output_folder = os.path.join(args.output_folder,args.process_fun)
    depth_bins = args.depth_bins
    num_bins = args.num_bins
    png_files = [f for f in os.listdir(depth_folder) if f.endswith('.png')]
    
    dataset = DepthProcessingDataset(
        png_files,
        depth_folder,
        output_folder,
        args,
        depth_bins,
        num_bins
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=False
    )
    
    # 使用并行处理
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for png_file in tqdm(png_files, desc="Processing images"):
            future = executor.submit(
                process_single_image,
                args,
                depth_bins,
                num_bins,
                depth_folder,
                output_folder,
                png_file
            )
            futures.append(future)

        # 等待所有任务完成
        for future in futures:
            future.result()