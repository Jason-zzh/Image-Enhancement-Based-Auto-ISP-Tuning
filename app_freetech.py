from flask import Flask, render_template, request, jsonify, send_from_directory
import torch
import numpy as np
import os
import rawpy
import imageio
import cv2  # For demosaic in fallback
from model_freetech import RawFormer
from load_dataset import load_data_MCR, load_data_SID  # If needed

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Model load (same)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RawFormer(dim=32).to(device)
checkpoint_path = './model_best.pth'
# checkpoint_path = 'result/SID/weights/model_best.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}, strict=True)
model.eval()
print('Model loaded.')

# def read_raw_file(raw_path, width, height, header_size=0):
#     """
#     Reads custom .raw file (e.g., header + uint16 Bayer data).
#     Returns (H, W) numpy array.
#     """
#     try:
#         with open(raw_path, 'rb') as f:
#             f.seek(header_size)
#             # 假设数据是 16-bit 无符号整数
#             raw_data = np.fromfile(f, dtype=np.uint16) 
        
#         expected_size = width * height
#         # 裁剪或调整大小以匹配预期
#         if raw_data.size < expected_size:
#              raise ValueError(f"Raw data size mismatch: expected {expected_size}, got {raw_data.size}")
        
#         raw_data = raw_data[:expected_size]
#         return raw_data.reshape(height, width)
#     except Exception as e:
#         print(f"Error reading custom raw file {raw_path}: {e}")
        # return None

def load_single_custom_raw(raw_path, width=1920, height=1280, header_size=0, bits=16):
    """
    Custom function to load, preprocess, and convert a single raw file for inference.
    Replaces rawpy logic with the custom 'read_raw_file'.
    
    Args:
        short_path (str): Path to the custom .raw file.
        raw_width/height/header_size (int): Configuration for the custom raw file.
        ap (int): Exposure scale factor (100 or 300).
        patch_size (int, optional): Size for center cropping.
        
    Returns:
        tuple: (input_raw_tensor (1, H, W), raw_data_np (H, W) for reference)
    """
    with open(raw_path, 'rb') as f:
        f.seek(header_size)
        raw_data = np.fromfile(f, dtype=np.uint16)

    expected_size = width * height
    raw_data = raw_data[:expected_size]
    raw_img = raw_data.reshape(height, width)

    # max_value = (1 << bits) - 1  # set bit here to 16
    # raw_normalized = raw_img / max_value 
    ap = 100
    raw_processed = (np.maximum(raw_img.astype(np.float32) - 512, 0) / (16383 - 512)) * ap
        
    # 转换为 PyTorch 张量
    input_raw = torch.from_numpy(raw_processed).float().unsqueeze(0) # (1, H, W)
    print(f"Loaded {raw_path}: Shape {input_raw.shape}")
    
    # 返回张量和原始数据（用于后续检查或可视化）
    return input_raw

def raw_to_srgb_skip_header(raw_path, width, height, bits, bayer_pattern='BGGR', header_size=0):
    """
    Load custom .raw and convert to sRGB (as before, but now with saving option).
    Returns the image array; caller must save if needed.
    """
    with open(raw_path, 'rb') as f:
        f.seek(header_size)
        raw_data = np.fromfile(f, dtype=np.uint16)

    expected_size = width * height
    raw_data = raw_data[:expected_size]
    raw_img = raw_data.reshape(height, width)

    max_value = (1 << bits) - 1  # set bit here to 16
    raw_normalized = raw_img / max_value 

    # 去马赛克
    if bayer_pattern == 'BGGR':
        # 转换为16位用于去马赛克处理
        rgb = cv2.cvtColor((raw_normalized * 65535).astype(np.uint16), cv2.COLOR_BAYER_BGGR2RGB)
    else:
        raise ValueError("仅支持BGGR拜耳格式")
    
    # 归一化到0-1范围
    rgb_normalized = rgb / 65535.0

    # Gamma校正（标准SRGB转换）
    def linear_to_srgb(linear):
        srgb = np.where(
            linear <= 0.0031308,
            linear * 12.92,
            1.055 * (linear ** (1/2.2)) - 0.055
        )
        return np.clip(srgb, 0.0, 1.0)
    
    # 转换为8位SRGB图像
    srgb_img = (linear_to_srgb(rgb_normalized) * 255).astype(np.uint8)
    return srgb_img

def save_and_get_url(img_array, base_name, prefix='orig'):
    """Helper: Save ndarray to static and return timestamped URL."""
    img_path = f'static/{prefix}_{base_name}.bmp'
    imageio.imwrite(img_path, img_array)
    import time
    timestamp = int(time.time())
    return f'/{img_path}?v={timestamp}'

def get_trad_preview(raw_path, width=1920, height=1280, bits=16, header_size=0):
    """Generate traditional preview using raw_to_srgb_skip_header and save URL."""
    try:
        srgb_img = raw_to_srgb_skip_header(
            raw_path=raw_path,
            width=width,
            height=height,
            bits=bits,
            header_size=header_size
        )
        base_name = os.path.splitext(os.path.basename(raw_path))[0]
        return save_and_get_url(srgb_img, base_name, 'trad')
    except Exception as e:
        print(f"Trad preview error: {e}")
        return None

def run_enhance(raw_path):
    """Patch-based enhance for large images: divide, process, stitch."""
    input_raw = load_single_custom_raw(raw_path)
    input_raw = input_raw.unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_rgb = model(input_raw)
        pred_rgb = (torch.clamp(pred_rgb, 0, 1).cpu().numpy().squeeze().transpose((1,2,0))*255).astype(np.uint8)
    
    base_name = os.path.splitext(os.path.basename(raw_path))[0]
    enhanced_path = f'static/enhanced_{base_name}.bmp'
    imageio.imwrite(enhanced_path, pred_rgb)
    return f'/{enhanced_path}'


# Routes (same)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and file.filename.lower().endswith(('.arw', '.dng', '.raw')):
        raw_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(raw_path)
        
        try:
            # Generate traditional preview (for custom .raw; adjust header_size if needed)
            trad_url = get_trad_preview(raw_path, width=1920, height=1280, bits=16, header_size=0)
            
            # Note: For standard RAW (.ARW/.DNG), you may want to add rawpy-based preview here
            # e.g., if rawpy succeeds in load_single_short_raw, use postprocess
            # For now, assuming custom .raw; extend if needed
            
            return jsonify({
                'success': True,
                # 'trad_url': trad_url,  # Or 'orig_url' if using rawpy
                'orig_url': trad_url,
                'message': 'Original preview ready! Click Run for enhancement.'
            })
        except Exception as e:
            print(f"Upload error: {e}")
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Invalid file type. Upload RAW (.ARW, .DNG, .RAW)'}), 400

@app.route('/enhance', methods=['POST'])
def enhance_file():
    data = request.json
    filename = data.get('filename')
    if not filename:
        return jsonify({'error': 'No filename'}), 400
    
    raw_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(raw_path):
        return jsonify({'error': 'File not found'}), 400
    
    try:
        enhanced_url = run_enhance(raw_path)
        return jsonify({
            'success': True,
            'enhanced_url': enhanced_url,
            'message': 'Enhancement complete!'
        })
    except Exception as e:
        print(f"Enhance error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)