import os
import io
import base64
import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify
from PIL import Image
from transformers import AutoTokenizer

app = Flask(__name__)

# --- 1. 配置路径与训练常数 ---
MODEL_PATH = "/mnt/oss/model.onnx"
TOKENIZER_PATH = "./model_config" 

# 填入你提取的缩放常数
DURATION_MEAN = 51878.51790957135
DURATION_STD = 288957.28163692716
PLAY_MEAN = 11.6972455308114
PLAY_STD = 2.893345257919881

# 加载模型和分词器
# 提示：如果是在 CPU 环境，使用 CPUExecutionProvider
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

def preprocess_image(image_base64):
    """
    对应训练代码中的 transforms.Compose 逻辑
    """
    try:
        img_data = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(img_data)).convert('RGB')
        
        # 1. Resize 224x224
        img = img.resize((224, 224))
        
        # 2. ToTensor: /255.0
        img_np = np.array(img).astype(np.float32) / 255.0
        
        # 3. Normalize: (val - mean) / std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np - mean) / std
        
        # 4. HWC -> CHW
        img_np = np.transpose(img_np, (2, 0, 1)) 
        return np.expand_dims(img_np, axis=0).astype(np.float32)
    except:
        # 如果图片处理失败，返回一个全零张量，防止报错
        return np.zeros((1, 3, 224, 224), dtype=np.float32)

@app.route('/predict', methods=['POST'])
@app.route('/invoke', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        
        # 1. 处理文本 (严格遵循训练代码的拼接方式)
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
        img_np = preprocess_image(data.get('image_base64', ''))
        
        # 3. 处理数值特征 (tabular) - 核心修正
        # 这里的 duration_sec 应该是原始秒数，play_log 应该是原始播放量数字
        duration_raw = float(data.get('duration_sec', 0))
        play_raw = float(data.get('play_log', 0))
        
        # 先取 Log1p (与训练逻辑一致)
        play_log_val = np.log1p(play_raw)
        
        # 再进行 StandardScaler 缩放
        duration_scaled = (duration_raw - DURATION_MEAN) / DURATION_STD
        play_log_scaled = (play_log_val - PLAY_MEAN) / PLAY_STD
        
        tab_np = np.array([[duration_scaled, play_log_scaled]], dtype=np.float32)

        # 4. 构建输入字典
        # 注意：这里的 key 必须与你导出 ONNX 时定义的 input_names 一致
        inputs = {
            'image': img_np,
            'input_ids': tokens['input_ids'].astype(np.int64),
            'attention_mask': tokens['attention_mask'].astype(np.int64),
            'tabular': tab_np
        }

        # 执行推理
        outputs = session.run(None, inputs)
        
        # 5. 结果处理
        # outputs[0] 的形状是 [1, 3]，代表三个类别的得分
        logits = outputs[0]
        pred = int(np.argmax(logits))
        
        # 可选：计算简单的概率（如果需要显示置信度）
        # exp_logits = np.exp(logits - np.max(logits))
        # probs = exp_logits / np.sum(exp_logits)
        # confidence = float(np.max(probs))

        return jsonify({
            'score': pred, 
            'status': 'success'
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e), 'status': 'fail'}), 400

if __name__ == '__main__':
    # Serverless 环境通常监听 9000 端口
    app.run(host='0.0.0.0', port=9000)
