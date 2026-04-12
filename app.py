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
@app.route('/invoke', methods=['POST']) # 新增：适配阿里云函数计算的默认调用路径
def predict():
    try:
        data = request.get_json()
        
        # 1. 处理文本数据
        # 假设输入包含 keyword 和 title
        text = f"[KW]{data.get('keyword', '')}[TTL]{data.get('title', '')}"
        tokens = tokenizer(
            text, 
            max_length=64, 
            padding='max_length', 
            truncation=True, 
            return_tensors='np'
        )
        
        # 2. 处理图像数据
        img_np = preprocess_image(data['image_base64'])
        
        # 3. 处理数值/结构化数据 (强制转为 float32)
        # 确保传入的是数字，增加默认值防止报错
        duration = float(data.get('duration_sec', 0))
        play_log = float(data.get('play_log', 0))
        tab_np = np.array([[duration, play_log]], dtype=np.float32)

        # 4. 构建 ONNX 推理输入
        # 显式使用 .astype 确保类型与模型定义严格一致
        inputs = {
            'image': img_np.astype(np.float32),
            'input_ids': tokens['input_ids'].astype(np.int64),
            'attention_mask': tokens['attention_mask'].astype(np.int64),
            'tabular': tab_np.astype(np.float32)
        }
        
        # 5. 执行推理
        outputs = session.run(None, inputs)
        
        # 6. 获取结果 (假设模型输出是分类的 Logits)
        pred = int(np.argmax(outputs[0]))
        # 如果模型输出是概率或单个分数，根据实际情况调整：
        # score = float(outputs[0][0]) 
        
        return jsonify({
            'score': pred, 
            'status': 'success'
        })
        
    except Exception as e:
        # 打印错误栈到云端日志，方便调试
        print(f"Error: {str(e)}")
        return jsonify({
            'error': str(e), 
            'status': 'fail'
        }), 400

if __name__ == '__main__':
    # 阿里云函数计算 (FC) 默认监听 9000 端口
    # 确保在 FC 控制台配置的监听端口与此一致
    app.run(host='0.0.0.0', port=9000)
