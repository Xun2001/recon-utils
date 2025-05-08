from func_utils.mesh_cpu import *
# from func_utils.mesh_gpu import *
from func_utils.mesh_w_time import *
import argparse
from tqdm import tqdm
import os

# 解析命令行参数
parser = argparse.ArgumentParser(description='深度图插值脚本')
parser.add_argument('--input_folder',"-i", type=str, default="/home/qinllgroup/hongxiangyu/git_project/livo2-data-utils/9-ImMesh/data/demo/images")
parser.add_argument('--output_folder',"-o", type=str, default="/home/qinllgroup/hongxiangyu/git_project/livo2-data-utils/9-ImMesh/data/demo/debug")
parser.add_argument('--max_depth', type=float, default=30, help='最大深度')
parser.add_argument('--max_edge', type=float, default=10, help='最大允许三角形边长（像素）')
parser.add_argument('--show', action='store_true', help='是否保存中间结果')
parser.add_argument('--bin_interval', type=float, default=2, help='分 BIN 的间距')
parser.add_argument('--process_fun', type=str, default='stack_acc', help='bin or stack')
args = parser.parse_args()

# 参数配置
depth_bins = np.arange(0.01, args.max_depth + 0.51, args.bin_interval)
num_bins = len(depth_bins) - 1

# 遍历输入文件夹中的PNG图片
input_folder = args.input_folder
output_folder = args.output_folder
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 输出当前主要参数
print(f"参数配置：")
print(f"最大深度：{args.max_depth}")
print(f"最大允许三角形边长（像素）：{args.max_edge}")
print(f"分 BIN 的间距：{args.bin_interval}")
print(f"处理方式：{args.process_fun}")

IS_CPU = True
png_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
for png_file in tqdm(png_files, desc="处理图片"):
    image_path = os.path.join(input_folder, png_file)
    save_path = os.path.join(output_folder,args.process_fun)
    # bin 之间间隔的大小对速度并无影响，说明主要影响剖分的是 点的数量，点的数量越多，时间越久
    # if args.process_fun == 'stack':
    #     process_image_stack_points(args,depth_bins,num_bins,image_path, save_path, os.path.splitext(png_file)[0])  # 102s/it 67.72s/it
    # elif args.process_fun == 'stack2':
    #     process_image_stack2_points(args,depth_bins,num_bins,image_path, save_path, os.path.splitext(png_file)[0]) # 69.8s/it 46.84s/it
    # elif args.process_fun == 'bin':
    #     process_image_bin_points(args,depth_bins,num_bins,image_path, save_path, os.path.splitext(png_file)[0])  # 
    process_image_stack_acc_points_time(args,depth_bins,num_bins,image_path, save_path, os.path.splitext(png_file)[0])