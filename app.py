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

# --- 建议把 Cookie 放在环境变量里，或者直接写在下面 ---
BILI_COOKIES = "buvid3=05F65891-9A80-CEF4-573C-34AC1E741ECC29339infoc; b_nut=1774677229; _uuid=D99DE5B7-8284-54A8-E2F5-FED78EFA1049F87220infoc; buvid_fp=e69b60c0497bf8fe189300b89fc45adf; buvid4=9F87B1B8-6501-C98A-2293-019138D8EFE130544-026032813-Dc4XJZfzpAB1Hw92vpu41g%3D%3D; rpdid=|(k)YmRY)uJR0J'u~~RY~~kRl; bili_jct=b4f8a9f9c5894c44c546fc964c385565; DedeUserID=3546563771107975; DedeUserID__ckMd5=10d7f5633d94eada; theme-tip-show=SHOWED; theme-avatar-tip-show=SHOWED; CURRENT_QUALITY=80; bp_t_offset_3546563771107975=1189881063988527104; CURRENT_FNVAL=4048; bsource=search_bing; bmg_af_switch=1; bmg_src_def_domain=i0.hdslb.com; bili_ticket=eyJhbGciOiJIUzI1NiIsImtpZCI6InMwMyIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NzY4MTY4ODUsImlhdCI6MTc3NjU1NzYyNSwicGx0IjotMX0.oCVdlhbd-z7gtvG7BEnCxvk9DHp97SpQO5UtD4gSVbo; bili_ticket_expires=1776816825; home_feed_column=4; browser_resolution=796-794; b_lsid=E1CBDA55_19DA318A2FB"

@app.route('/get_video_link', methods=['POST'])
def get_video_link():
    try:
        data = request.get_json(force=True)
        bvid = data.get('bvid')
        video_url = f"https://www.bilibili.com/video/{bvid}"
        
        ydl_opts = {
            'format': 'best[ext=mp4]/best',
            'quiet': True,
            'no_warnings': True,
            'nocheckcertificate': True,
            # 这里的 http_headers 非常关键
            'http_headers': {
                'Cookie': BILI_COOKIES,
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Referer': 'https://www.bilibili.com/',
                'Origin': 'https://www.bilibili.com/',
                'Accept': '*/*',
            }
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            direct_url = info.get('url')
            return jsonify({
                'status': 'success',
                'direct_url': direct_url,
                'title': info.get('title')
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
