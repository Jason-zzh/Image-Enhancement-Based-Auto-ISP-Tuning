from flask import Flask, request, jsonify, Response
import torch
import numpy as np
import os
import cv2
from model_freetech import RawFormer
import threading
import queue
import time

app = Flask(__name__)

# Model load (same)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RawFormer(dim=32).to(device)
checkpoint_path = 'result/Freetech/weights/model_best.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}, strict=True)
model.eval()
print('Model loaded.')

# 全局变量 for streaming
frame_queue = queue.Queue(maxsize=5)  # 捕获到处理 (RAW bytes)
enhanced_queue = queue.Queue(maxsize=5)  # 增强RGB to stream
orig_queue = queue.Queue(maxsize=5)  # 原RGB to stream (for contrast)
capture_thread = None
process_thread = None
streaming_active = False

def load_single_custom_raw_from_bytes(raw_bytes, width=1920, height=1280, header_size=0, bits=16):
    """
    预设为RGB输入: 假设raw_bytes是BGR frame bytes (from cv2.read()), 
    但为兼容RAW, 先试RAW加载; 如果失败, 视作RGB并demosaic/skip.
    提醒: 如果板子输出RGB (非RAW), 注释掉RAW部分, 直接return RGB tensor (1,3,H,W) 但模型需改输入通道.
    """
    try:
        # 预设尝试RAW加载 (默认)
        raw_data = np.frombuffer(raw_bytes[header_size:], dtype=np.uint16)[:width*height]
        raw_img = raw_data.reshape(height, width).astype(np.float32)
        ap = 100
        raw_processed = (np.maximum(raw_img - 512, 0) / (16383 - 512)) * ap
        input_tensor = torch.from_numpy(raw_processed).float().unsqueeze(0)  # (1,1,H,W) for RAW model
        is_raw = True
    except:
        # Fallback to RGB (假设bytes是BGR frame, e.g., height*width*3 uint8)
        frame_bgr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(height, width, 3)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        input_tensor = torch.from_numpy(frame_rgb).permute(2,0,1).unsqueeze(0).float()  # (1,3,H,W)
        # RawFormer是for 1-ch RAW; 如用RGB, 需改model forward或加demosaic层. 暂预设RAW, 测试后调.
        is_raw = False
        print("Fallback to RGB input - model may need adjustment!")
    
    print(f"Loaded frame: Shape {input_tensor.shape}, RAW={is_raw}")
    return input_tensor

def raw_to_rgb_from_bytes(raw_bytes, width=1920, height=1280, bits=16, header_size=0, bayer_pattern='BGGR'):
    """
    转换bytes到RGB for original stream. 预设RAW, fallback RGB.
    """
    try:
        # 预设RAW
        raw_data = np.frombuffer(raw_bytes[header_size:], dtype=np.uint16)[:width*height]
        raw_img = raw_data.reshape(height, width)
        max_value = (1 << bits) - 1
        raw_normalized = raw_img / max_value 
        rgb = cv2.cvtColor((raw_normalized * 65535).astype(np.uint16), cv2.COLOR_BAYER_BGGR2RGB)
        rgb_normalized = rgb / 65535.0
        # Gamma to sRGB
        def linear_to_srgb(linear):
            srgb = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * (linear ** (1/2.2)) - 0.055)
            return np.clip(srgb, 0.0, 1.0)
        srgb_img = (linear_to_srgb(rgb_normalized) * 255).astype(np.uint8)
        orig_bgr = cv2.cvtColor(srgb_img, cv2.COLOR_RGB2BGR)
    except:
        # Fallback: 假设RGB bytes
        orig_bgr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(height, width, 3)
    
    return orig_bgr

def capture_frames(source_url='udp://192.168.2.1:5000'):
    """线程: 从板子流捕获帧bytes"""
    cap = cv2.VideoCapture(source_url)
    if not cap.isOpened():
        print(f"Error opening stream: {source_url}")
        return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 低延迟
    cap.set(cv2.CAP_PROP_FPS, 30)  # 预设30fps
    while streaming_active:
        ret, frame = cap.read()  # frame: BGR np.array
        if ret:
            frame_bytes = frame.tobytes()  # 存bytes for queue
            if not frame_queue.full():
                frame_queue.put(frame_bytes)
            # 同时推原流 (转换到RGB for orig_queue)
            orig_bgr = raw_to_rgb_from_bytes(frame_bytes)  # 或直接frame if RGB
            if not orig_queue.full():
                orig_queue.put(orig_bgr)
        time.sleep(0.033)  # ~30fps
    cap.release()
    print("Capture thread stopped.")

def process_frames(width=1920, height=1280):
    """线程: 处理RAW → 增强RGB"""
    while streaming_active:
        if not frame_queue.empty():
            raw_bytes = frame_queue.get()
            input_raw = load_single_custom_raw_from_bytes(raw_bytes, width, height).to(device)
            with torch.no_grad():
                pred_rgb = model(input_raw)
                pred_rgb = torch.clamp(pred_rgb, 0, 1).cpu().numpy().squeeze().transpose((1,2,0)) * 255
                enhanced_bgr = cv2.cvtColor(pred_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
            if not enhanced_queue.full():
                enhanced_queue.put(enhanced_bgr)
        time.sleep(0.01)  # 低CPU

def generate_mjpeg(queue, boundary='frame'):
    """通用MJPEG generator from queue"""
    while streaming_active:
        if not queue.empty():
            frame = queue.get()
            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                yield (b'--' + boundary.encode() + b'\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.033)

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head><title>Real-time RAW Enhancer</title></head>
    <body>
        <h1>Real-time Stream Enhancement</h1>
        <input type="text" id="sourceUrl" value="udp://192.168.2.1:5000" placeholder="Stream URL">
        <button onclick="startStream()">Start Stream</button>
        <button onclick="stopStream()">Stop Stream</button>
        <div style="display:flex;">
            <div>
                <h3>Original</h3>
                <img id="origVideo" src="" style="width:50%; height:auto;">
            </div>
            <div>
                <h3>Enhanced</h3>
                <img id="enhVideo" src="" style="width:50%; height:auto;">
            </div>
        </div>

        <script>
        function startStream() {
            const url = document.getElementById('sourceUrl').value;
            fetch('/start_stream', {method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({source_url: url})})
                .then(r => r.json()).then(data => {
                    if (data.success) {
                        document.getElementById('origVideo').src = '/orig_stream';
                        document.getElementById('enhVideo').src = '/enh_stream';
                    } else {
                        alert(data.error);
                    }
                });
        }
        function stopStream() {
            fetch('/stop_stream', {method: 'POST'}).then(() => {
                document.getElementById('origVideo').src = '';
                document.getElementById('enhVideo').src = '';
            });
        }
        </script>
    </body>
    </html>
    '''

# 新增 streaming 路由
@app.route('/start_stream', methods=['POST'])
def start_stream():
    global streaming_active, capture_thread, process_thread
    if streaming_active:
        return jsonify({'error': 'Already running'}), 400
    data = request.json or {}
    source_url = data.get('source_url', 'udp://192.168.2.1:5000')  # 默认板子UDP
    width, height = 1920, 1280  # 预设分辨率
    streaming_active = True
    capture_thread = threading.Thread(target=capture_frames, args=(source_url,))
    process_thread = threading.Thread(target=process_frames, args=(width, height))
    capture_thread.start()
    process_thread.start()
    # 预热模型
    dummy = torch.zeros(1, 1, height, width).to(device)  # 假设RAW 1ch
    with torch.no_grad():
        _ = model(dummy)
    print(f"Stream started: {source_url}")
    return jsonify({'success': True, 'message': 'Stream started'})

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    global streaming_active
    streaming_active = False
    if capture_thread:
        capture_thread.join(timeout=2)
    if process_thread:
        process_thread.join(timeout=2)
    print("Stream stopped.")
    return jsonify({'success': True})

@app.route('/orig_stream')
def orig_stream():
    if not streaming_active:
        return 'Stream not active', 404
    return Response(generate_mjpeg(orig_queue, 'origframe'), mimetype='multipart/x-mixed-replace; boundary=origframe')

@app.route('/enh_stream')
def enh_stream():
    if not streaming_active:
        return 'Stream not active', 404
    return Response(generate_mjpeg(enhanced_queue, 'enhframe'), mimetype='multipart/x-mixed-replace; boundary=enhframe')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002, threaded=True)