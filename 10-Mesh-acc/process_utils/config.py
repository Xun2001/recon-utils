import argparse
import numpy as np

def mesh_config():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='深度图插值脚本')
    parser.add_argument('--input_folder',"-i", type=str, default="/home/qinllgroup/hongxiangyu/git_project/livo2-data-utils/10-Mesh-acc/data/tree_01_frame_10/depth_maps")
    parser.add_argument('--output_folder',"-o", type=str, default="/home/qinllgroup/hongxiangyu/git_project/livo2-data-utils/10-Mesh-acc/data/tree_01_frame_10/depth_meshs")
    parser.add_argument('--max_depth', type=float, default=30, help='最大深度')
    parser.add_argument('--max_edge', type=float, default=10, help='最大允许三角形边长（像素）')
    parser.add_argument('--show', action='store_true', help='是否保存中间结果')
    parser.add_argument('--bin_interval', type=float, default=1, help='分 BIN 的间距')
    parser.add_argument('--bin_workers', type=int, default=15, help='每张图片并行处理bin的数量')
    parser.add_argument('--num_image_workers', type=int, default=6, help='每次处理的图片数量')
    
    parser.add_argument('--num_bins', type=int, default=15, help='num_bins')
    parser.add_argument('--process_fun', type=str, default='stack_acc', help='bin or stack')
    
    args = parser.parse_args()
    # assert args.num_bins == args.bin_workers 
    # 参数配置
    # args.depth_bins = np.arange(0.01, args.max_depth + 0.51, args.bin_interval)
    # args.num_bins = len(args.depth_bins) - 1
    
    # 输出当前主要参数
    print(f"参数配置：")
    print(f"最大深度：{args.max_depth}")
    print(f"最大允许三角形边长（像素）：{args.max_edge}")
    # print(f"分 BIN 的间距：{args.bin_interval}")
    print(f"分 BIN 的数量：{args.num_bins}")
    print(f"处理方式：{args.process_fun}")
    print(f"bin_workers: {args.bin_workers}")
    print(f"num_image_workers: {args.num_image_workers}")

    return args

def mesh_config_image():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='深度图插值脚本')
    parser.add_argument('--input_folder',"-i", type=str, default="/home/qinllgroup/hongxiangyu/git_project/livo2-data-utils/10-Mesh-acc/data/tree_01_mini_20/depth_maps")
    parser.add_argument('--output_folder',"-o", type=str, default="/home/qinllgroup/hongxiangyu/git_project/livo2-data-utils/10-Mesh-acc/data/tree_01_mini_20/depth_mesh")
    parser.add_argument('--max_depth', type=float, default=30, help='最大深度')
    parser.add_argument('--max_edge', type=float, default=10, help='最大允许三角形边长（像素）')
    parser.add_argument('--show', action='store_true', help='是否保存中间结果')
    parser.add_argument('--bin_interval', type=float, default=1, help='分 BIN 的间距')
    parser.add_argument('--bin_workers', type=int, default=15, help='每张图片并行处理bin的数量')
    parser.add_argument('--num_image_workers', type=int, default=1, help='每次处理的图片数量')
    
    # parser.add_argument('--num_bins', type=int, default=20, help='num_bins')
    parser.add_argument('--process_fun', type=str, default='stack_acc', help='bin or stack')
    
    args = parser.parse_args()
    # assert args.num_bins == args.bin_workers 
    # 参数配置
    args.depth_bins = np.arange(0.01, args.max_depth + 0.51, args.bin_interval)
    args.num_bins = len(args.depth_bins) - 1
    
    # 输出当前主要参数
    print(f"参数配置：")
    print(f"最大深度：{args.max_depth}")
    print(f"最大允许三角形边长（像素）：{args.max_edge}")
    # print(f"分 BIN 的间距：{args.bin_interval}")
    print(f"分 BIN 的数量：{args.num_bins}")
    print(f"处理方式：{args.process_fun}")
    print(f"bin_workers: {args.bin_workers}")
    print(f"num_image_workers: {args.num_image_workers}")

    return args
