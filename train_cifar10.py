import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import time

print("=" * 50)
print("项目：CIFAR-10 图像分类训练 (使用真实数据)")
print("=" * 50)

# 1. 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 2. 数据增强与加载 (针对CIFAR-10)
print("正在加载CIFAR-10数据集（你已下载，将直接解压使用）...")
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 关键：root='./data' 会指向你下载好的数据集位置
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)
print(f"训练集大小: {len(trainset)}，测试集大小: {len(testset)}")

# 3. 构建模型 (ResNet-18，适配CIFAR-10)
print("正在构建ResNet-18模型...")
model = resnet18(pretrained=True)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
model.fc = nn.Linear(512, 10)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 4. 训练与测试函数
def train_one_epoch(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if i % 100 == 99:
            print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss/100:.3f}, Acc: {100.*correct/total:.2f}%')
            running_loss = 0.0
            correct = 0
            total = 0
    print(f'Epoch {epoch+1} 训练完成')

def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    acc = 100. * correct / total
    print(f'测试集准确率: {acc:.2f}%')
    return acc

# 5. 开始训练
print("开始训练（约20-30分钟，电脑自动跑）...")
for epoch in range(20):
    train_one_epoch(epoch)
    scheduler.step()
    if (epoch + 1) % 5 == 0:
        test()

print("=" * 50)
print("训练完成！最终测试结果：")
final_acc = test()

# 6. 保存模型与导出ONNX
torch.save(model.state_dict(), 'cifar10_resnet18.pth')
print("模型权重已保存为 cifar10_resnet18.pth")

dummy_input = torch.randn(1, 3, 32, 32).to(device)
torch.onnx.export(model, dummy_input, "cifar10_resnet18.onnx",
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
print("✅ ONNX导出成功！文件：cifar10_resnet18.onnx")
print("=" * 50)