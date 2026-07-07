# AI 项目集：从模型训练到部署

这个仓库包含了我完成的两个AI入门项目，涵盖了**计算机视觉**和**数据挖掘**两个方向的核心流程，并实现了模型的工程化部署。

---
markdown
## 📁 项目列表

### 1. CIFAR-10 图像分类与模型部署 (ResNet-18)
*技术栈：Python · PyTorch · ResNet-18 · ONNX · Flask*

这是一个完整的图像分类落地项目。我训练了一个ResNet-18深度卷积神经网络来识别CIFAR-10彩色图像，并将模型部署为HTTP API服务。

- **模型表现**：在测试集上达到 **94.44%** 的准确率
- **核心功能**：用户上传一张图片，API返回Top-3的类别预测及置信度
- **部署方式**：PyTorch模型 → ONNX格式 → ONNX Runtime推理 → Flask封装的RESTful API
- **关键文件**：[`train_cifar10.py`](./train_cifar10.py) (训练脚本) | [`app.py`](./app.py) (部署服务)

### 2. 电商用户复购预测
*技术栈：Python · Pandas · XGBoost · Scikit-learn*

这是一个典型的机器学习二分类项目，模拟了电商场景下预测用户是否会再次购买的核心任务。

- **建模流程**：数据生成（模拟真实用户行为）→ 特征工程（构造12维特征）→ 模型训练（XGBoost）→ 模型评估（AUC/准确率）
- **业务洞察**：通过特征重要性分析，识别出“购买率”、“平均停留时长”等影响复购的关键因素，并输出可落地的营销建议
- **关键文件**：[`train_xgboost.py`](./train_xgboost.py) (完整建模流程)

---

## 🚀 如何运行项目

### 环境配置
1.  确保已安装Python (3.8或更高版本)
2.  安装依赖库（建议使用虚拟环境）：
    ```bash
    pip install torch torchvision onnx onnxruntime flask pillow pandas numpy scikit-learn xgboost

### 运行图像分类服务

在项目目录下，确保有 `fashion_mnist_cnn.onnx` 模型文件，执行以下命令启动服务：

```bash
python app.py
```

服务启动后，可通过 `POST` 请求 `http://localhost:5000/predict` 上传图片进行测试。

---

### 运行复购预测模型

在项目目录下直接运行训练脚本：

```bash
python train_xgboost.py
```

## 📂 项目结构

```
├── app.py                 # Flask部署服务 (图像分类API)
├── train_cifar10.py       # CIFAR-10图像分类模型训练脚本 (ResNet-18)
├── train_xgboost.py       # 复购预测模型训练脚本
├── cifar10_resnet18.onnx  # 训练好的ONNX模型文件
├── fashion_mnist_cnn.onnx # (可选) 旧版Fashion-MNIST模型
├── .gitignore             # Git忽略文件配置
└── README.md              # 项目说明文档
```

---

## 📫 关于我

- **GitHub**: [baozhenmao](https://github.com/baozhenmao)
- **邮箱**: Maobaozhen.iot@QQ.com

*—— 持续学习与实践的AI初学者，正在向AI软件开发工程师的目标前进。*
