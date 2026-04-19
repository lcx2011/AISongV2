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

@app.route('/get_video_link', methods=['POST'])
def get_video_link():
    try:
        data = request.get_json(force=True)
        bvid = data.get('bvid')
        
        video_url = f"https://www.bilibili.com/video/{bvid}"
        
        # 🆕 增强版配置，专门应对 412 错误
        ydl_opts = {
            'format': 'best[ext=mp4]/best',
            'quiet': True,
            'no_warnings': True,
            # 伪装成真实的 Chrome 浏览器
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'http_headers': {
                'Referer': 'https://www.bilibili.com/',
                'Origin': 'https://www.bilibili.com/',
                'Accept-Language': 'zh-CN,zh;q=0.9',
            },
            # 强制不检查 SSL 证书（有时能解决握手问题）
            'nocheckcertificate': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # 尝试获取
            info = ydl.extract_info(video_url, download=False)
            return jsonify({
                'status': 'success',
                'direct_url': info.get('url'),
                'title': info.get('title')
            })
    except Exception as e:
        error_msg = str(e)
        # 如果还是 412，说明 B 站彻底封锁了该机房 IP，需要提示用户提供 Cookie
        if "412" in error_msg:
            return jsonify({
                'status': 'fail', 
                'msg': "B站拦截了云端请求(412)。建议在服务端配置Cookie或更换函数计算区域。"
            }), 412
        return jsonify({'status': 'fail', 'msg': error_msg}), 500
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
