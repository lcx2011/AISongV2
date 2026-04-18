import os
import io
import base64
import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify
from PIL import Image
from transformers import AutoTokenizer

app = Flask(__name__)

# --- 1. 配置 (根据实际情况修改) ---
# 如果挂载了 OSS，路径通常是 /mnt/oss/model_v2.onnx
MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/oss/model_v2.onnx")
TOKENIZER_PATH = "./model_config" 

# 【关键】填入你第一步在 Kaggle 打印出的常数
DURATION_MEAN = 20013.62078774617
DURATION_STD = 178296.57879180563
PLAY_MEAN = 11.235865505828247
PLAY_STD = 3.037800872598104

# 初始化模型（启动时加载一次）
print(f"Loading model from {MODEL_PATH}...")
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

def preprocess_image(image_base64):
    try:
        if not image_base64: return np.zeros((1, 3, 224, 224), dtype=np.float32)
        # 解码 Base64
        img_data = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(img_data)).convert('RGB')
        img = img.resize((224, 224))
        
        # 归一化
        img_np = np.array(img).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np - mean) / std
        img_np = np.transpose(img_np, (2, 0, 1)) # HWC -> CHW
        return np.expand_dims(img_np, axis=0).astype(np.float32)
    except:
        return np.zeros((1, 3, 224, 224), dtype=np.float32)

@app.route('/invoke', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        
        # 1. 文本预处理 (严格匹配训练逻辑: [KW]xxx [TTL]xxx)
        kw = str(data.get('keyword', ''))
        ttl = str(data.get('title', ''))
        text = f"[KW]{kw} [TTL]{ttl}"
        
        tokens = tokenizer(
            text, 
            max_length=96, 
            padding='max_length', 
            truncation=True, 
            return_tensors='np'
        )
        
        # 2. 图像预处理
        img_np = preprocess_image(data.get('image_base64', ''))
        
        # 3. 数值预处理
        d_raw = float(data.get('duration_sec', 0))
        p_raw = float(data.get('play', 0))
        
        # 对应训练逻辑: log1p -> scale
        p_log = np.log1p(p_raw)
        d_scaled = (d_raw - DURATION_MEAN) / DURATION_STD
        p_scaled = (p_log - PLAY_MEAN) / PLAY_STD
        tab_np = np.array([[d_scaled, p_scaled]], dtype=np.float32)

        # 4. ONNX 推理
        inputs = {
            'image': img_np,
            'input_ids': tokens['input_ids'].astype(np.int64),
            'attention_mask': tokens['attention_mask'].astype(np.int64),
            'tabular': tab_np
        }
        outputs = session.run(None, inputs)
        logits = outputs[0]
        
        # 5. 后处理
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
    # 阿里云 FC 默认端口
    app.run(host='0.0.0.0', port=9000)
