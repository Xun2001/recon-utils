from process_utils.process_depth import *
from process_utils.config import mesh_config
from process_utils.process_cupy import process_image_stack_cupy_points_time
import argparse
from tqdm import tqdm
import os



if __name__ == '__main__':
    """
    cd ~/git_project/livo2-data-utils/10-Mesh-acc/
    conda activate ~/envs/env_mesh
    python demo-acc.py --max_edge 10 --bin_interval 2

    """
    args = mesh_config()
    
    # 遍历输入文件夹中的PNG图片
    input_folder = args.input_folder
    
    save_path = os.path.join(args.output_folder,f"{args.process_fun}_{int(args.max_edge)}_{int(args.bin_interval)}")
    os.makedirs(save_path, exist_ok=True)
    print(f"输出保存的文件夹: {save_path}")
    # png_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    png_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')], key=lambda x: int(''.join(filter(str.isdigit, x))))
    for png_file in tqdm(png_files, desc="处理图片"):
        image_path = os.path.join(input_folder, png_file)

        # useful function: process_image_stack_acc_points_time(args, image_path, save_path, os.path.splitext(png_file)[0])
        process_image_stack_pall_points_time(args, image_path, save_path, os.path.splitext(png_file)[0])