from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import onnxruntime as ort
import numpy as np

app = Flask(__name__)

# 加载ONNX模型（使用CIFAR-10模型）
session = ort.InferenceSession('cifar10_resnet18.onnx')

# CIFAR-10的预处理：和训练时保持一致
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# CIFAR-10的10个类别
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'CIFAR-10 图像分类服务已启动',
        'usage': 'POST /predict 上传图片',
        'classes': classes
    })

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': '请上传图片，字段名称为 image'}), 400
    
    file = request.files['image']
    try:
        # 读取并预处理图片
        img = Image.open(io.BytesIO(file.read()))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_tensor = transform(img).unsqueeze(0).numpy()
        
        # ONNX推理
        outputs = session.run(['output'], {'input': img_tensor})[0]
        probs = torch.softmax(torch.tensor(outputs[0]), dim=0).numpy()
        
        # 取Top-3
        top3_idx = probs.argsort()[-3:][::-1]
        result = {
            'predictions': [
                {'class': classes[i], 'confidence': float(probs[i])} 
                for i in top3_idx
            ]
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'图片处理失败: {str(e)}'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
