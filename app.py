import os
import io
import base64
import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify
from PIL import Image
from transformers import AutoTokenizer

app = Flask(__name__)

# 1. 加载配置和模型
# 注意：确保路径正确
MODEL_PATH = "/mnt/oss/model.onnx"
TOKENIZER_PATH = "./model_config"

session = ort.InferenceSession(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

def preprocess_image(image_base64):
    img_data = base64.b64decode(image_base64)
    img = Image.open(io.BytesIO(img_data)).convert('RGB')
    img = img.resize((224, 224))
    img_np = np.array(img).astype('float32') / 255.0
    # 标准化 (与训练一致)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = (img_np - mean) / std
    img_np = np.transpose(img_np, (2, 0, 1)) # HWC -> CHW
    return np.expand_dims(img_np, axis=0)

@app.route('/predict', methods=['POST'])
@app.route('/invoke', methods=['POST'])  # 新增这一行
def predict():
    try:
        data = request.get_json(force=True, silent=True)

        if data is None:
            return jsonify({'error': 'No JSON data received', 'status': 'fail'}), 400
        
        # 处理文本
        text = f"[KW]{data['keyword']}[TTL]{data['title']}"
        tokens = tokenizer(text, max_length=64, padding='max_length', truncation=True, return_tensors='np')
        
        # 处理图像
        img_np = preprocess_image(data['image_base64'])
        
        # 处理数值 (假设你已经缩放过或传入原始值)
        tab_np = np.array([[data['duration_sec'], data['play_log']]], dtype=np.float32)

        # ONNX 推理
        inputs = {
            'image': img_np,
            'input_ids': tokens['input_ids'].astype(np.int64),
            'attention_mask': tokens['attention_mask'].astype(np.int64),
            'tabular': tab_np
        }
        outputs = session.run(None, inputs)
        pred = int(np.argmax(outputs[0]))
        
        return jsonify({'score': pred, 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'fail'})

if __name__ == '__main__':
    # 函数计算默认监听 9000 端口
    app.run(host='0.0.0.0', port=9000)
