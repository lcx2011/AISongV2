import os
import io
import base64
import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify
from PIL import Image
from transformers import AutoTokenizer

app = Flask(__name__)

# 1. 配置路径
MODEL_PATH = "/mnt/oss/model.onnx"
TOKENIZER_PATH = "./model_config" # 确保该目录下有 macbert 的 vocab.txt 等文件

# 加载模型和分词器
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

def preprocess_image(image_base64):
    """
    对应训练代码中的 transforms.Compose 逻辑
    """
    img_data = base64.b64decode(image_base64)
    img = Image.open(io.BytesIO(img_data)).convert('RGB')
    
    # 训练代码中提前 Resize 到了 224x224
    img = img.resize((224, 224))
    
    # ToTensor() 效果: /255.0 并转为 CHW
    img_np = np.array(img).astype(np.float32) / 255.0
    
    # Normalize 逻辑
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = (img_np - mean) / std
    
    img_np = np.transpose(img_np, (2, 0, 1)) # HWC -> CHW
    return np.expand_dims(img_np, axis=0).astype(np.float32)

@app.route('/predict', methods=['POST'])
@app.route('/invoke', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        
        # 1. 处理文本 (严格遵循训练代码的拼接方式)
        # text_input = "[KW]" + keyword + "[TTL]" + title
        kw = str(data.get('keyword', ''))
        ttl = str(data.get('title', ''))
        text = f"[KW]{kw}[TTL]{ttl}"
        
        tokens = tokenizer(
            text, 
            max_length=64, 
            padding='max_length', 
            truncation=True, 
            return_tensors='np'
        )
        
        # 2. 处理图像
        img_np = preprocess_image(data['image_base64'])
        
        # 3. 处理数值特征 (tabular)
        # 注意：训练代码中使用了 StandardScaler 对 duration_sec 和 play_log 进行了缩放
        # 如果追求准确度，这里应该使用训练时保存的 scaler 进行 transform
        duration = float(data.get('duration_sec', 0))
        play_log = np.log1p(float(data.get('play_log', 0))) # 训练代码中有 np.log1p 逻辑
        
        # 临时处理：这里假设你传入的是原始值，理想做法是加载 scaler.pkl
        tab_np = np.array([[duration, play_log]], dtype=np.float32)

        # 4. 构建输入字典 (严格区分数据类型)
        inputs = {
            'image': img_np.astype(np.float32),               # 图像必须是 float32
            'input_ids': tokens['input_ids'].astype(np.int64),       # 文本 ID 必须是 int64
            'attention_mask': tokens['attention_mask'].astype(np.int64), # 遮罩必须是 int64
            'tabular': tab_np.astype(np.float32)              # 数值特征必须是 float32
        }

        # 执行推理
        outputs = session.run(None, inputs)
        
        # 结果处理
        pred = int(np.argmax(outputs[0]))
        
        return jsonify({
            'score': pred, 
            'status': 'success'
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e), 'status': 'fail'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000)
