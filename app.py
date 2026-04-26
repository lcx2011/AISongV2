import os
import io
import numpy as np
import onnxruntime as ort
import requests
from flask import Flask, request, jsonify
from PIL import Image
from transformers import AutoTokenizer
import yt_dlp

app = Flask(__name__)

# --- 配置 ---
MODEL_RANK_PATH = os.environ.get("MODEL_PATH", "/mnt/oss/model_v2.onnx")
MODEL_SUB_PATH = "/mnt/oss/subtitle_model.onnx" # 确保此路径有字幕模型
TOKENIZER_PATH = "./model_config" 
BILI_COOKIES = os.environ.get("BILI_COOKIES", "")
COOKIE_FILE_PATH = "/tmp/bilibili_cookies.txt"

def create_cookie_file(cookie_str, filepath):
    if not cookie_str: return
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("# Netscape HTTP Cookie File\n")
        domain = ".bilibili.com"
        for item in cookie_str.split(';'):
            if '=' not in item: continue
            key, value = item.strip().split('=', 1)
            f.write(f"{domain}\tTRUE\t/\tFALSE\t1893456000\t{key}\t{value}\n")

create_cookie_file(BILI_COOKIES, COOKIE_FILE_PATH)

# 初始化两个 AI 模型
print("Loading Models...")
sess_rank = ort.InferenceSession(MODEL_RANK_PATH, providers=['CPUExecutionProvider'])
sess_sub = ort.InferenceSession(MODEL_SUB_PATH, providers=['CPUExecutionProvider'])
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

DURATION_MEAN, DURATION_STD = 20013.62078774617, 178296.57879180563
PLAY_MEAN, PLAY_STD = 11.235865505828247, 3.037800872598104

req_session = requests.Session()
req_session.headers.update({'User-Agent': 'Mozilla/5.0...'})

@app.route('/invoke', methods=['GET', 'POST'])
def keep_warm(): return "OK", 200

@app.route('/get_video_link', methods=['POST'])
def get_video_link():
    try:
        data = request.get_json(force=True)
        bvid = data.get('bvid')
        ydl_opts = {
            'format': 'bestvideo[vcodec^=avc1]+bestaudio[ext=m4a]/best',
            'cookiefile': COOKIE_FILE_PATH if os.path.exists(COOKIE_FILE_PATH) else None,
            'quiet': True
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"https://www.bilibili.com/video/{bvid}", download=False)
            formats = info.get('requested_formats')
            v_url, a_url = (formats[0].get('url'), formats[1].get('url')) if formats else (info.get('url'), None)
            return jsonify({'status': 'success', 'video_url': v_url, 'audio_url': a_url, 'title': info.get('title')})
    except Exception as e:
        return jsonify({'status': 'fail', 'msg': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        text = f"[KW]{data.get('keyword','')} [TTL]{data.get('title','')}"
        tokens = tokenizer(text, max_length=96, padding='max_length', truncation=True, return_tensors='np')
        
        # 图片预处理
        pic_url = data.get('pic_url', '')
        if pic_url.startswith('//'): pic_url = "https:" + pic_url
        img_resp = req_session.get(pic_url + "@320w_200h_1e_1c.jpg", timeout=5)
        img = Image.open(io.BytesIO(img_resp.content)).convert('RGB').resize((224, 224))
        img_np = (np.array(img).astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img_np = np.expand_dims(np.transpose(img_np, (2, 0, 1)), 0).astype(np.float32)
        
        d_raw, p_raw = float(data.get('duration_sec', 0)), float(data.get('play', 0))
        tab_np = np.array([[(d_raw - DURATION_MEAN)/DURATION_STD, (np.log1p(p_raw) - PLAY_MEAN)/PLAY_STD]], dtype=np.float32)
        
        inputs = {'image': img_np, 'input_ids': tokens['input_ids'].astype(np.int64), 
                  'attention_mask': tokens['attention_mask'].astype(np.int64), 'tabular': tab_np}
        logits = sess_rank.run(None, inputs)[0]
        pred_idx = int(np.argmax(logits))
        return jsonify({'score': pred_idx, 'label': ["不适合", "一般", "推荐"][pred_idx], 'status': 'success'})
    except Exception as e: return jsonify({'error': str(e), 'status': 'fail'}), 400

@app.route('/predict_subtitle', methods=['POST'])
def predict_subtitle():
    """接收多张图片进行字幕批量检测"""
    try:
        files = request.files.getlist("images")
        results = []
        for file in files:
            img = Image.open(file.stream).convert('RGB').resize((224, 224))
            img_np = (np.array(img).astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            img_np = np.expand_dims(np.transpose(img_np, (2, 0, 1)), 0).astype(np.float32)
            
            logits = sess_sub.run(None, {'input': img_np})[0]
            results.append(int(np.argmax(logits))) # 1表示有字幕
        return jsonify({'status': 'success', 'preds': results})
    except Exception as e: return jsonify({'status': 'fail', 'msg': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000)
