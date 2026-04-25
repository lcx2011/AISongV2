import os
import io
import base64
import numpy as np
import onnxruntime as ort
import requests
from flask import Flask, request, jsonify
from PIL import Image
from transformers import AutoTokenizer
import yt_dlp

app = Flask(__name__)

# --- 配置 ---
MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/oss/model_v2.onnx")
TOKENIZER_PATH = "./model_config" 
# 填入你的 B 站 Cookie
BILI_COOKIES = os.environ.get("BILI_COOKIES", "")
# ！！！新增：动态生成标准 Cookie 文件供 yt-dlp 使用 ！！！
COOKIE_FILE_PATH = "/tmp/bilibili_cookies.txt"

def create_cookie_file(cookie_str, filepath):
    """将普通的 Cookie 字符串转换为 yt-dlp 支持的 Netscape 格式文件"""
    if not cookie_str:
        return
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("# Netscape HTTP Cookie File\n")
        # yt-dlp 识别 B 站的关键域名
        domain = ".bilibili.com"
        for item in cookie_str.split(';'):
            if '=' not in item:
                continue
            key, value = item.strip().split('=', 1)
            # 格式: 域名, 是否包含子域, 路径, 是否HTTPS, 过期时间(这里写死2030年), 键, 值
            f.write(f"{domain}\tTRUE\t/\tFALSE\t1893456000\t{key}\t{value}\n")

# 在服务启动时生成 cookie 文件
create_cookie_file(BILI_COOKIES, COOKIE_FILE_PATH)
# 初始化 AI 模型
print("Loading AI Model...")
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

DURATION_MEAN, DURATION_STD = 20013.62078774617, 178296.57879180563
PLAY_MEAN, PLAY_STD = 11.235865505828247, 3.037800872598104

# 全局 Requests Session，用于服务端高速拉取封面
req_session = requests.Session()
req_session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'https://www.bilibili.com/'
})
# --- 接口 0: 定时保活/健康检查 ---
@app.route('/health', methods=['GET'])
def health():
    # 只要这个接口被访问，整个容器（包括已经加载好的模型）就会被平台保留
    return "OK", 200

# --- 接口 1: 高清解析 ---
@app.route('/get_video_link', methods=['POST'])
def get_video_link():
    try:
        data = request.get_json(force=True)
        bvid = data.get('bvid')
        video_url = f"https://www.bilibili.com/video/{bvid}"
        
        ydl_opts = {
            'format': 'bestvideo+bestaudio/best',
            'quiet': True,
            'no_warnings': True,
            'nocheckcertificate': True,
            # 关键修改 1：使用 cookiefile 替代 http_headers 里的 Cookie
            'cookiefile': COOKIE_FILE_PATH if os.path.exists(COOKIE_FILE_PATH) else None,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Referer': 'https://www.bilibili.com/',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
                'Accept-Language': 'zh-CN,zh;q=0.9',
                # 删除了这里的 'Cookie': BILI_COOKIES
            },
            # 如果使用 file cookie 依然报错，尝试注释掉下面这一行
            'extractor_args': {'bilibili': {'web_client': 'web'}},
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            formats = info.get('requested_formats')
            
            if formats and len(formats) >= 2:
                v_url, a_url = formats[0].get('url'), formats[1].get('url')
            else:
                v_url, a_url = info.get('url'), None
                
            return jsonify({
                'status': 'success',
                'video_url': v_url,
                'audio_url': a_url,
                'title': info.get('title')
            })
    except Exception as e:
        return jsonify({'status': 'fail', 'msg': str(e)}), 500
# --- 接口 2: AI 预测 ---
def fetch_and_preprocess_image(pic_url):
    """服务端直接从B站拉取图片并转为模型所需格式"""
    try:
        if not pic_url:
            return np.zeros((1, 3, 224, 224), dtype=np.float32)
            
        # 补全 URL 协议
        if pic_url.startswith('//'):
            pic_url = "https:" + pic_url
            
        # 请求 B 站缩略图 (极大提升服务端拉取速度和内存效率)
        if "@" not in pic_url:
            pic_url = f"{pic_url}@320w_200h_1e_1c.jpg"
            
        img_resp = req_session.get(pic_url, timeout=5)
        img_resp.raise_for_status()
        
        img = Image.open(io.BytesIO(img_resp.content)).convert('RGB').resize((224, 224))
        img_np = np.array(img).astype(np.float32) / 255.0
        mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
        img_np = np.transpose((img_np - mean) / std, (2, 0, 1))
        return np.expand_dims(img_np, axis=0).astype(np.float32)
    except Exception as e:
        print(f"Image fetch error: {e}")
        return np.zeros((1, 3, 224, 224), dtype=np.float32)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        text = f"[KW]{data.get('keyword','')} [TTL]{data.get('title','')}"
        tokens = tokenizer(text, max_length=96, padding='max_length', truncation=True, return_tensors='np')
        
        # 改动：调用服务端拉取图片函数
        img_np = fetch_and_preprocess_image(data.get('pic_url', ''))
        
        d_raw, p_raw = float(data.get('duration_sec', 0)), float(data.get('play', 0))
        tab_np = np.array([[(d_raw - DURATION_MEAN)/DURATION_STD, (np.log1p(p_raw) - PLAY_MEAN)/PLAY_STD]], dtype=np.float32)
        
        inputs = {'image': img_np, 'input_ids': tokens['input_ids'].astype(np.int64), 
                  'attention_mask': tokens['attention_mask'].astype(np.int64), 'tabular': tab_np}
        logits = session.run(None, inputs)[0]
        probs = np.exp(logits - np.max(logits)) / np.sum(np.exp(logits - np.max(logits)))
        pred_idx = int(np.argmax(logits))
        res_map = ["不适合", "一般", "很适合(推荐)"]
        return jsonify({'score': pred_idx, 'label': res_map[pred_idx], 'confidence': float(np.max(probs)), 'status': 'success'})
    except Exception as e: 
        return jsonify({'error': str(e), 'status': 'fail'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000)
