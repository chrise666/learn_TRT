import os
import time
import torch
import torchvision
import torchvision.transforms as transforms
from classify.model.custom_model import resnet50


device = "cuda" if torch.cuda.is_available() else "cpu"
model = resnet50(num_classes=10).to(device)
model.load_state_dict(torch.load('resnet50_fp32.pth'))

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=128, shuffle=False, num_workers=0
)

# 模型评估函数
def evaluate_model(model, test_loader, device="cpu"):
    model.eval()
    correct = 0
    total = 0
    inference_time = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            start_time = time.time()
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            inference_time += time.time() - start_time

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    avg_inference_time = inference_time / len(test_loader.dataset) * 1000  # 毫秒/样本

    return accuracy, avg_inference_time

# 评估模型
print("Evaluating Original Model...")
original_accuracy, original_inf_time = evaluate_model(model, test_loader, device)  
fp32_size = os.path.getsize('resnet50_fp32.pth') / 1024**2  # MB

# 分析模型效果
print(f"Original Model Accuracy: {original_accuracy:.2f}%")
print(f"Original Model Size: {fp32_size:.2f} MB")
print(f"Original Model Inference Time: {original_inf_time:.4f} ms per sample")