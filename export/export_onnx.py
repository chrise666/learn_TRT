import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 添加项目根目录到Python路径

import torch
from develop.classify.model.custom_model import * 

os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

# 实例化模型
model = resnet50(num_classes=3)
model.load_state_dict(torch.load('E:/workspace/learn_AI/save/shape_classification/best.pth'))
# model = torch.load('E:/workspace/save/shape_classification/best.pt')
# model = SimpleCNN()
model.eval()  # 若存在batchnorm、dropout层则一定要eval()!!!!再export

dummy_input = torch.randn(1, 3, 128, 128)
onnx_path = '../save/shape_classification/resnet50.onnx'
input_names = ["input"]
output_names = ["output"]

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
print("ONNX模型导出成功")
