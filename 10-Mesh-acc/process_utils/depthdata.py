from torch.utils.data import Dataset, DataLoader
from .process_depth import *

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

    def __getitem__(self, idx):
        png_file = self.file_list[idx]
        image_path = os.path.join(self.input_folder, png_file)
        save_path = os.path.join(self.output_folder, self.args.process_fun)
        os.makedirs(save_path, exist_ok=True)
        
        # 调用原始处理函数
        process_depth_map_single(
            self.args,
            self.depth_bins,
            self.num_bins,
            image_path,
            save_path,
            os.path.splitext(png_file)[0]
        )
        return 0  # 返回空值