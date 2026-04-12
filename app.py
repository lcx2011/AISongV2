import os
import io
import base64
import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify
from PIL import Image
from transformers import AutoTokenizer

app = Flask(__name__)

# 加载模型和分词器
MODEL_PATH = "/mnt/oss/model.onnx"
TOKENIZER_PATH = "./model_config"

# 显式指定使用 CPU 运行
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

def preprocess_image(image_base64):
    """处理图像并确保输出为 float32"""
    img_data = base64.b64decode(image_base64)
    img = Image.open(io.BytesIO(img_data)).convert('RGB')
    img = img.resize((224, 224))
    
    img_np = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = (img_np - mean) / std
    img_np = np.transpose(img_np, (2, 0, 1))
    return np.expand_dims(img_np, axis=0).astype(np.float32)

@app.route('/predict', methods=['POST'])
@app.route('/invoke', methods=['POST'])
def predict():
    try:
        # 使用 force=True 解决 Content-Type 引起的 415 错误
        data = request.get_json(force=True)
        
        # 1. 处理文本
        text = f"[KW]{data.get('keyword', '')}[TTL]{data.get('title', '')}"
        tokens = tokenizer(text, max_length=64, padding='max_length', truncation=True, return_tensors='np')
        
        # 2. 处理图像
        img_np = preprocess_image(data['image_base64'])
        
        # 3. 处理数值特征
        tab_np = np.array([[
            float(data.get('duration_sec', 0)), 
            float(data.get('play_log', 0))
        ]], dtype=np.float32)

        # 4. 构建输入 (解决 Expected: float 错误的关键点)
        # 报错说收到 int64，说明模型要求 float。
        # 我们把 input_ids 和 attention_mask 全部强制转为 float32
        inputs = {
            'image': img_np.astype(np.float32),
            'input_ids': tokens['input_ids'].astype(np.float32),       # 修改点：从 int64 转为 float32
            'attention_mask': tokens['attention_mask'].astype(np.float32), # 修改点：从 int64 转为 float32
            'tabular': tab_np.astype(np.float32)
        }

        # 执行推理
        outputs = session.run(None, inputs)
        
        # 获取结果
        pred = int(np.argmax(outputs[0]))
        
        return jsonify({
            'score': pred, 
            'status': 'success'
        })
        
    except Exception as e:
        # 在 FC 日志中打印详细信息
        print(f"Prediction Error: {str(e)}")
        return jsonify({
            'error': str(e), 
            'status': 'fail'
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000)
