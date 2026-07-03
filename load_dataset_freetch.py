import rawpy
from torch.utils.data import Dataset
import tqdm
import random
import imageio
import numpy as np
import torch
import cv2
import os
import glob

def read_raw_file(raw_path, width, height, header_size=0, bits=16):
    """
    读取RAW文件
    """
    try:
        with open(raw_path, 'rb') as f:
            f.seek(header_size)
            raw_data = np.fromfile(f, dtype=np.uint16)
        
        expected_size = width * height
        
        # 检查数据大小
        if len(raw_data) < expected_size:
            print(f"Warning: RAW文件数据不足，期望{expected_size}，实际{len(raw_data)}")
            raw_data = np.resize(raw_data, expected_size)
        elif len(raw_data) > expected_size:
            raw_data = raw_data[:expected_size]
        
        return raw_data.reshape(height, width)
    
    except Exception as e:
        print(f"读取RAW文件失败 {raw_path}: {e}")
        return np.zeros((height, width), dtype=np.uint16)

class load_data_RawRGB_Paired(Dataset):
    """
    从指定路径加载RAW和RGB配对数据的Dataset
    路径结构:
        root_path/
            RAW/    # 存放.raw文件
            RGB/    # 存放.rgb文件
    """
    def __init__(self, root_path, 
                 raw_width=1920, raw_height=1280, raw_header_size=0,
                 rgb_width=1920, rgb_height=1280,
                 patch_size=512, training=True, 
                 normalize_range=(-1, 1)):   # 归一化范围
        
        self.training = training
        self.patch_size = patch_size
        self.normalize_range = normalize_range
        
        # 配置参数
        self.raw_width = raw_width
        self.raw_height = raw_height
        self.raw_header_size = raw_header_size
        self.rgb_width = rgb_width
        self.rgb_height = rgb_height
        
        # 扫描文件路径
        self.raw_rgb_pairs = self._scan_paired_files(root_path)
        
        if len(self.raw_rgb_pairs) == 0:
            raise ValueError(f"在路径 {root_path} 下未找到配对的RAW和RGB文件")
        
        print(f'\n数据加载完成，共找到{len(self.raw_rgb_pairs)}对RAW-RGB样本 ......\n')

    def _scan_paired_files(self, root_path):
        """扫描配对的RAW和RGB文件"""
        raw_dir = os.path.join(root_path, 'RAW')
        rgb_dir = os.path.join(root_path, 'RGB')
        
        if not os.path.exists(raw_dir):
            raise ValueError(f"RAW目录不存在: {raw_dir}")
        if not os.path.exists(rgb_dir):
            raise ValueError(f"RGB目录不存在: {rgb_dir}")
        
        # 获取所有RAW文件
        raw_files = glob.glob(os.path.join(raw_dir, '*.raw'))
        raw_files.sort()
        
        pairs = []
        
        for raw_path in tqdm.tqdm(raw_files, desc="扫描配对文件"):
            # 获取文件名（不含扩展名）
            base_name = os.path.splitext(os.path.basename(raw_path))[0]
            
            # 查找对应的RGB文件
            rgb_path = os.path.join(rgb_dir, base_name + '.rgb')
            if not os.path.exists(rgb_path):
                rgb_path = os.path.join(rgb_dir, base_name + '.bmp')  # 也尝试.bmp格式
                
            if os.path.exists(rgb_path):
                pairs.append((raw_path, rgb_path))
            else:
                print(f"警告: 未找到 {base_name} 对应的RGB文件")
        
        return pairs

    def _read_rgb_file(self, rgb_path):
        """读取RGB文件"""
        try:
            # 根据文件扩展名选择读取方式
            if rgb_path.lower().endswith('.rgb'):
                # 读取原始RGB数据
                with open(rgb_path, 'rb') as f:
                    rgb_data = np.fromfile(f, dtype=np.uint8)
                
                expected_size = self.rgb_width * self.rgb_height * 3
                if len(rgb_data) < expected_size:
                    print(f"Warning: RGB文件数据不足，期望{expected_size}，实际{len(rgb_data)}")
                    rgb_data = np.resize(rgb_data, expected_size)
                elif len(rgb_data) > expected_size:
                    rgb_data = rgb_data[:expected_size]
                
                rgb_array = rgb_data.reshape(self.rgb_height, self.rgb_width, 3)
                return rgb_array
            else:
                # 使用imageio读取其他格式（bmp, jpg, png等）
                rgb_array = imageio.imread(rgb_path)
                if len(rgb_array.shape) == 2:  # 灰度图转RGB
                    rgb_array = cv2.cvtColor(rgb_array, cv2.COLOR_GRAY2RGB)
                elif rgb_array.shape[2] == 4:  # RGBA转RGB
                    rgb_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGBA2RGB)
                
                # 调整尺寸
                if rgb_array.shape[0] != self.rgb_height or rgb_array.shape[1] != self.rgb_width:
                    rgb_array = cv2.resize(rgb_array, (self.rgb_width, self.rgb_height))
                
                return rgb_array
                
        except Exception as e:
            print(f"读取RGB文件失败 {rgb_path}: {e}")
            return np.zeros((self.rgb_height, self.rgb_width, 3), dtype=np.uint8)

    def _normalize_rgb(self, rgb_array):
        """归一化RGB数据到指定范围"""
        rgb_float = rgb_array.astype(np.float32)
        
        if self.normalize_range == (-1, 1):
            # 归一化到 [-1, 1]
            return (rgb_float - 127.5) / 127.5
        elif self.normalize_range == (0, 1):
            # 归一化到 [0, 1]
            return rgb_float / 255.0
        else:
            # 不归一化
            return rgb_float

    def __len__(self):
        return len(self.raw_rgb_pairs)

    def __getitem__(self, idx):
        raw_path, rgb_path = self.raw_rgb_pairs[idx]
        
        # 动态读取文件
        raw_img = read_raw_file(
            raw_path=raw_path,
            width=self.raw_width,
            height=self.raw_height,
            header_size=self.raw_header_size
        )
        
        rgb_img = self._read_rgb_file(rgb_path)

        H, W = raw_img.shape

        if self.training and self.patch_size > 0:
            # 训练时随机裁剪
            i = random.randint(0, max(0, H - self.patch_size - 2)) // 2 * 2
            j = random.randint(0, max(0, W - self.patch_size - 2)) // 2 * 2

            raw_crop = raw_img[i:i+self.patch_size, j:j+self.patch_size]
            rgb_crop = rgb_img[i:i+self.patch_size, j:j+self.patch_size, :]

            # 数据增强
            if random.random() > 0.5:
                raw_crop = np.fliplr(raw_crop).copy()
                rgb_crop = np.fliplr(rgb_crop).copy()

            if random.random() < 0.2:
                raw_crop = np.flipud(raw_crop).copy()
                rgb_crop = np.flipud(rgb_crop).copy()
        else:
            raw_crop = raw_img
            rgb_crop = rgb_img

        # RAW预处理 (保持你的原始逻辑)
        ap = 100
        raw_processed = (np.maximum(raw_crop.astype(np.float32) - 512, 0) / (16383 - 512)) * ap

        # RGB处理
        rgb_processed = self._normalize_rgb(rgb_crop)

        # 转换为Tensor
        raw_tensor = torch.from_numpy(raw_processed).float().unsqueeze(0)  # (1, H, W)
        rgb_tensor = torch.from_numpy(np.transpose(rgb_processed, (2, 0, 1))).float()  # (3, H, W)

        return raw_tensor, rgb_tensor

    def get_file_info(self, idx):
        """获取文件信息"""
        raw_path, rgb_path = self.raw_rgb_pairs[idx]
        return {
            'raw_file': os.path.basename(raw_path),
            'rgb_file': os.path.basename(rgb_path),
            'raw_path': raw_path,
            'rgb_path': rgb_path
        }

# 使用示例
if __name__ == "__main__":
    # 示例: 从指定路径加载数据
    root_path = "./freetech_dataset"  # 替换为你的数据路径
    
    dataset = load_data_RawRGB_Paired(
        root_path=root_path,
        raw_width=1920, raw_height=1280,
        rgb_width=1920, rgb_height=1280,
        patch_size=512,
        training=True,
        normalize_range=(0, 1)  # 输出范围[-1, 1]
    )
    
    # 测试数据加载
    raw_tensor, rgb_tensor = dataset[0]
    print(f"RAW tensor shape: {raw_tensor.shape}")   # (1, 512, 512)
    print(f"RGB tensor shape: {rgb_tensor.shape}")   # (3, 512, 512)
    print(f"RGB range: [{rgb_tensor.min():.3f}, {rgb_tensor.max():.3f}]")  # [-1, 1]
    
    # 查看文件信息
    file_info = dataset.get_file_info(0)
    print(f"RAW文件: {file_info['raw_file']}")
    print(f"RGB文件: {file_info['rgb_file']}")