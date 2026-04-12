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
# 注意：确保 OSS 挂载路径正确，或者模型已打包在镜像/代码包中
MODEL_PATH = "/mnt/oss/model.onnx"
TOKENIZER_PATH = "./model_config"

# 初始化 ONNX Runtime 会话
# 如果是在没有 GPU 的 Serverless 环境，使用 CPUExecutionProvider
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

def preprocess_image(image_base64):
    """
    处理图像：Base64 解码 -> 调整大小 -> 归一化 -> 转换为 Float32
    """
    img_data = base64.b64decode(image_base64)
    img = Image.open(io.BytesIO(img_data)).convert('RGB')
    img = img.resize((224, 224))
    
    # 1. 转换为 float32 数组并除以 255.0
    img_np = np.array(img).astype(np.float32) / 255.0
    
    # 2. 标准化 (必须显式指定 dtype=np.float32，否则 NumPy 会默认使用 float64)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = (img_np - mean) / std
    
    # 3. HWC 转 CHW (Height, Width, Channel -> Channel, Height, Width)
    img_np = np.transpose(img_np, (2, 0, 1))
    
    # 4. 增加 Batch 维度 (C, H, W -> 1, C, H, W)
    return np.expand_dims(img_np, axis=0).astype(np.float32)

@app.route('/predict', methods=['POST'])
@app.route('/invoke', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        
        # 1. 处理文本
        text = f"[KW]{data.get('keyword', '')}[TTL]{data.get('title', '')}"
        tokens = tokenizer(text, max_length=64, padding='max_length', truncation=True, return_tensors='np')
        
        # 2. 处理图像
        img_np = preprocess_image(data['image_base64'])
        
        # 3. 处理数值 (强制转换，确保万无一失)
        # 如果模型报错 Actual: int64, expected: float，极大概率是这里
        tab_np = np.array([[
            float(data.get('duration_sec', 0)), 
            float(data.get('play_log', 0))
        ]], dtype=np.float32)

        # 4. ONNX 推理
        # 注意：这里我们增加了对 tokens 的类型检查
        # 如果报错依然存在，尝试把 .astype(np.int64) 改为 .astype(np.float32)
        inputs = {
            'image': img_np.astype(np.float32),
            'input_ids': tokens['input_ids'].astype(np.int64), # 大多数模型是 int64
            'attention_mask': tokens['attention_mask'].astype(np.int64), # 大多数模型是 int64
            'tabular': tab_np.astype(np.float32)
        }
        
        # --- 调试代码：查看模型到底想要什么类型 ---
        # for input_meta in session.get_inputs():
        #     print(f"Name: {input_meta.name}, Type: {input_meta.type}, Shape: {input_meta.shape}")
        # ---------------------------------------

        outputs = session.run(None, inputs)
        pred = int(np.argmax(outputs[0]))
        
        return jsonify({'score': pred, 'status': 'success'})
    except Exception as e:
        # 将具体的错误信息打印出来
        print(f"Exception details: {str(e)}")
        return jsonify({'error': str(e), 'status': 'fail'}), 400

if __name__ == '__main__':
    # 阿里云函数计算 (FC) 默认监听 9000 端口
    # 确保在 FC 控制台配置的监听端口与此一致
    app.run(host='0.0.0.0', port=9000)
