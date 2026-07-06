from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import onnxruntime as ort
import numpy as np

app = Flask(__name__)

# 加载ONNX模型
session = ort.InferenceSession('fashion_mnist_cnn.onnx')

# 预处理：和训练时保持一致（28x28灰度图，归一化）
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': '请上传图片，字段名称为 image'}), 400
    
    file = request.files['image']
    try:
        img = Image.open(io.BytesIO(file.read()))
        img_tensor = transform(img).unsqueeze(0).numpy()
        
        outputs = session.run(['output'], {'input': img_tensor})[0]
        probs = torch.softmax(torch.tensor(outputs[0]), dim=0).numpy()
        
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

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Fashion-MNIST 分类服务已启动', 'usage': 'POST /predict 上传图片'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)