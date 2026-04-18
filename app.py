import os
import io
import base64
import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify
from PIL import Image
from transformers import AutoTokenizer
import yt_dlp  # 🆕 新增

app = Flask(__name__)

# --- 1. 配置 ---
MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/oss/model_v2.onnx")
TOKENIZER_PATH = "./model_config" 

DURATION_MEAN = 20013.62078774617
DURATION_STD = 178296.57879180563
PLAY_MEAN = 11.235865505828247
PLAY_STD = 3.037800872598104

# 初始化
print(f"Loading model...")
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

# --- 🆕 2. 新增：云端解析视频直链接口 ---
@app.route('/get_video_link', methods=['POST'])
def get_video_link():
    try:
        data = request.get_json(force=True)
        bvid = data.get('bvid')
        if not bvid:
            return jsonify({'status': 'fail', 'msg': '缺少bvid'}), 400
            
        video_url = f"https://www.bilibili.com/video/{bvid}"
        
        # yt-dlp 配置：只找包含音轨的单个 mp4 文件 (通常最高 720p)
        # 这样客户端 Win7 就不需要 ffmpeg 进行合并了
        ydl_opts = {
            'format': 'best[ext=mp4]/best', 
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            direct_url = info.get('url')
            return jsonify({
                'status': 'success',
                'direct_url': direct_url,
                'title': info.get('title', 'video')
            })
    except Exception as e:
        return jsonify({'status': 'fail', 'msg': str(e)}), 500

# --- 3. 原有的 AI 预测接口 ---
def preprocess_image(image_base64):
    try:
        if not image_base64: return np.zeros((1, 3, 224, 224), dtype=np.float32)
        img_data = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(img_data)).convert('RGB')
        img = img.resize((224, 224))
        img_np = np.array(img).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np - mean) / std
        img_np = np.transpose(img_np, (2, 0, 1))
        return np.expand_dims(img_np, axis=0).astype(np.float32)
    except:
        return np.zeros((1, 3, 224, 224), dtype=np.float32)

@app.route('/invoke', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        kw = str(data.get('keyword', ''))
        ttl = str(data.get('title', ''))
        text = f"[KW]{kw} [TTL]{ttl}"
        
        tokens = tokenizer(text, max_length=96, padding='max_length', truncation=True, return_tensors='np')
        img_np = preprocess_image(data.get('image_base64', ''))
        
        d_raw = float(data.get('duration_sec', 0))
        p_raw = float(data.get('play', 0))
        p_log = np.log1p(p_raw)
        d_scaled = (d_raw - DURATION_MEAN) / DURATION_STD
        p_scaled = (p_log - PLAY_MEAN) / PLAY_STD
        tab_np = np.array([[d_scaled, p_scaled]], dtype=np.float32)

        inputs = {
            'image': img_np,
            'input_ids': tokens['input_ids'].astype(np.int64),
            'attention_mask': tokens['attention_mask'].astype(np.int64),
            'tabular': tab_np
        }
        outputs = session.run(None, inputs)
        logits = outputs[0]
        
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        pred_idx = int(np.argmax(logits))
        res_map = ["不适合", "一般", "很适合(推荐)"]
        
        return jsonify({
            'score': pred_idx,
            'label': res_map[pred_idx],
            'confidence': float(np.max(probs)),
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'fail'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000)
