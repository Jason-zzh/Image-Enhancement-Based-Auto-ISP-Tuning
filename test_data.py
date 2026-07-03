import rawpy
from torch.utils.data import Dataset
import tqdm
import random
import imageio.v2 as imageio
import numpy as np
import torch
import cv2
import os
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置matplotlib
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
mpl.rcParams['axes.unicode_minus'] = False

def read_raw_file(raw_path, width, height, header_size=56, bits=16):
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
    """
    def __init__(self, root_path, 
                 raw_width=1920, raw_height=1280, raw_header_size=56,
                 rgb_width=1920, rgb_height=1280,
                 patch_size=512, training=True, 
                 normalize_range=(-1, 1)):
        
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
        
        raw_files = glob.glob(os.path.join(raw_dir, '*.raw'))
        raw_files.sort()
        
        pairs = []
        
        for raw_path in tqdm.tqdm(raw_files, desc="扫描配对文件"):
            base_name = os.path.splitext(os.path.basename(raw_path))[0]
            
            rgb_path = os.path.join(rgb_dir, base_name + '.rgb')
            if not os.path.exists(rgb_path):
                rgb_path = os.path.join(rgb_dir, base_name + '.bmp')
                
            if os.path.exists(rgb_path):
                pairs.append((raw_path, rgb_path))
            else:
                print(f"警告: 未找到 {base_name} 对应的RGB文件")
        
        return pairs

    def _read_rgb_file(self, rgb_path):
        """读取RGB文件"""
        try:
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
                # 使用imageio.v2读取其他格式
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
        # 确保数据类型正确
        rgb_float = rgb_array.astype(np.float32)
        
        if self.normalize_range == (-1, 1):
            return (rgb_float - 127.5) / 127.5
        elif self.normalize_range == (0, 1):
            return rgb_float / 255.0
        else:
            return rgb_float

    def _denormalize_rgb(self, rgb_tensor):
        """将归一化的RGB tensor反归一化到[0, 255]范围"""
        if isinstance(rgb_tensor, torch.Tensor):
            rgb_array = rgb_tensor.cpu().numpy()
        else:
            rgb_array = rgb_tensor
            
        # 确保数据类型正确
        rgb_array = rgb_array.astype(np.float32)
            
        if rgb_array.ndim == 3 and rgb_array.shape[0] == 3:  # (C, H, W)
            rgb_array = np.transpose(rgb_array, (1, 2, 0))  # (H, W, C)
        
        if self.normalize_range == (-1, 1):
            rgb_array = (rgb_array * 127.5 + 127.5)
        elif self.normalize_range == (0, 1):
            rgb_array = (rgb_array * 255.0)
        
        # 确保在有效范围内并转换为uint8
        rgb_array = np.clip(rgb_array, 0, 255).astype(np.uint8)
        return rgb_array

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
            # 确保裁剪位置有效
            i = random.randint(0, max(0, H - self.patch_size))
            j = random.randint(0, max(0, W - self.patch_size))

            # 确保不越界
            i = min(i, H - self.patch_size)
            j = min(j, W - self.patch_size)

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

        # RAW预处理 - 修复数据类型问题
        raw_float = raw_crop.astype(np.float32)
        raw_processed = (np.maximum(raw_float - 512.0, 0.0) / (16383.0 - 512.0)) * 100.0

        # RGB处理
        rgb_processed = self._normalize_rgb(rgb_crop)

        # 转换为Tensor - 确保数据类型正确
        raw_tensor = torch.from_numpy(raw_processed).float().unsqueeze(0)  # (1, H, W)
        rgb_tensor = torch.from_numpy(rgb_processed).float()  # (H, W, 3)
        rgb_tensor = rgb_tensor.permute(2, 0, 1)  # (3, H, W)

        return raw_tensor, rgb_tensor

    def debug_sample(self, idx):
        """调试样本数据"""
        print(f"\n=== 调试样本 {idx} ===")
        
        raw_path, rgb_path = self.raw_rgb_pairs[idx]
        print(f"RAW文件: {raw_path}")
        print(f"RGB文件: {rgb_path}")
        
        # 读取原始数据
        raw_img = read_raw_file(raw_path, self.raw_width, self.raw_height, self.raw_header_size)
        rgb_img = self._read_rgb_file(rgb_path)
        
        print(f"RAW图像形状: {raw_img.shape}, 数据类型: {raw_img.dtype}, 范围: [{raw_img.min()}, {raw_img.max()}]")
        print(f"RGB图像形状: {rgb_img.shape}, 数据类型: {rgb_img.dtype}, 范围: [{rgb_img.min()}, {rgb_img.max()}]")
        
        # 获取处理后的tensor
        raw_tensor, rgb_tensor = self[idx]
        
        print(f"RAW tensor形状: {raw_tensor.shape}, 数据类型: {raw_tensor.dtype}, 范围: [{raw_tensor.min():.3f}, {raw_tensor.max():.3f}]")
        print(f"RGB tensor形状: {rgb_tensor.shape}, 数据类型: {rgb_tensor.dtype}, 范围: [{rgb_tensor.min():.3f}, {rgb_tensor.max():.3f}]")
        
        return raw_tensor, rgb_tensor

    def visualize_sample(self, idx, figsize=(15, 5), save_path=None):
        """可视化指定索引的样本"""
        try:
            print(f"正在可视化样本 {idx}...")
            
            # 先调试数据
            raw_tensor, rgb_tensor = self.debug_sample(idx)
            
            # 反归一化RGB tensor
            print("反归一化RGB数据...")
            rgb_display = self._denormalize_rgb(rgb_tensor)
            print(f"反归一化后RGB形状: {rgb_display.shape}, 数据类型: {rgb_display.dtype}, 范围: [{rgb_display.min()}, {rgb_display.max()}]")
            
            # 准备RAW图像显示
            print("准备RAW数据显示...")
            raw_display = raw_tensor.squeeze().cpu().numpy()
            raw_display = raw_display.astype(np.float32)  # 确保数据类型
            raw_display = (raw_display - raw_display.min()) / (raw_display.max() - raw_display.min() + 1e-8)
            print(f"RAW显示数据形状: {raw_display.shape}, 数据类型: {raw_display.dtype}, 范围: [{raw_display.min():.3f}, {raw_display.max():.3f}]")
            
            # 创建图像
            print("创建图像...")
            fig, axes = plt.subplots(1, 3, figsize=figsize)
            
            # 显示RAW图像
            print("显示RAW图像...")
            im0 = axes[0].imshow(raw_display, cmap='gray')
            axes[0].set_title(f'RAW Image\nShape: {raw_tensor.shape}')
            axes[0].axis('off')
            plt.colorbar(im0, ax=axes[0], fraction=0.046)
            
            # 显示RGB图像
            print("显示RGB图像...")
            im1 = axes[1].imshow(rgb_display)
            axes[1].set_title(f'RGB Image\nShape: {rgb_display.shape}')
            axes[1].axis('off')
            
            # 显示RGB tensor重建
            print("显示RGB tensor重建...")
            rgb_from_tensor = self._denormalize_rgb(rgb_tensor.cpu().numpy())
            im2 = axes[2].imshow(rgb_from_tensor)
            axes[2].set_title('RGB (from tensor)')
            axes[2].axis('off')
            
            try:
                plt.tight_layout()
            except:
                pass
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor='white')
                print(f"图像已保存到: {save_path}")
            
            plt.show()
            print("可视化完成!")
            
            return raw_tensor, rgb_tensor
            
        except Exception as e:
            print(f"可视化失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def simple_visualize(self, idx=0):
        """简化版可视化，使用OpenCV"""
        try:
            raw_tensor, rgb_tensor = self[idx]
            
            # 转换为numpy并反归一化
            rgb_np = self._denormalize_rgb(rgb_tensor)
            raw_np = raw_tensor.squeeze().cpu().numpy()
            raw_np = (raw_np - raw_np.min()) / (raw_np.max() - raw_np.min() + 1e-8)
            raw_np = (raw_np * 255).astype(np.uint8)
            
            print(f"RAW显示数据: {raw_np.shape}, {raw_np.dtype}, [{raw_np.min()}, {raw_np.max()}]")
            print(f"RGB显示数据: {rgb_np.shape}, {rgb_np.dtype}, [{rgb_np.min()}, {rgb_np.max()}]")
            
            # 使用OpenCV显示
            rgb_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
            
            # 调整大小便于显示
            display_size = (800, 600)
            rgb_display = cv2.resize(rgb_bgr, display_size)
            raw_display = cv2.resize(raw_np, display_size)
            
            # 显示图像
            cv2.imshow('RAW Image', raw_display)
            cv2.imshow('RGB Image', rgb_display)
            print("按任意键关闭窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # 同时保存图像文件
            cv2.imwrite('debug_raw.png', raw_display)
            cv2.imwrite('debug_rgb.png', rgb_display)
            print("图像已保存为 debug_raw.png 和 debug_rgb.png")
            
        except Exception as e:
            print(f"简化可视化失败: {e}")
            import traceback
            traceback.print_exc()

# 使用示例
if __name__ == "__main__":
    root_path = "./freetech_dataset"  # 替换为你的数据路径
    
    dataset = load_data_RawRGB_Paired(
        root_path=root_path,
        raw_width=1920, raw_height=1280,
        rgb_width=1920, rgb_height=1280,
        patch_size=512,
        training=True,
        normalize_range=(-1, 1)
    )
    
    # 先尝试简化版可视化
    print("=== 尝试简化版可视化 ===")
    dataset.simple_visualize(0)
    
    # 然后尝试完整版可视化
    print("\n=== 尝试完整版可视化 ===")
    dataset.visualize_sample(0)
    
    # 测试数据加载
    print("\n=== 数据加载测试 ===")
    raw_tensor, rgb_tensor = dataset[0]
    print(f"RAW tensor shape: {raw_tensor.shape}")
    print(f"RGB tensor shape: {rgb_tensor.shape}")