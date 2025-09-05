import torch
from classify.model.custom_model import resnet50
from train import prepare_qat_model, convert_pt2e
from torch.ao.quantization import move_exported_model_to_eval

# 创建基础模型
base_model = resnet50(num_classes=3)

# 准备QAT模型
model = prepare_qat_model(base_model, "E:/workspace/learn_AI/save/shape_classification/resnet50_int8.pth")

# 转换为量化模型
quantized_model = convert_pt2e(model, use_reference_representation=False, fold_quantize=True)
move_exported_model_to_eval(quantized_model)

# 创建示例输入（确保尺寸匹配训练时的设置）
dummy_input = (torch.randn(1, 3, 128, 128), )

model_exp = torch.export.export(quantized_model, dummy_input, strict=False)

# 导出为ONNX
try:
    torch.onnx.export(
        model_exp,               
        dummy_input,                   
        "E:/workspace/learn_AI/save/shape_classification/resnet50_quantized.onnx",     
        export_params=True,            
        opset_version=18,   
        optimize=True,           
        do_constant_folding=True,   
        external_data=False,   
        input_names=['input'],         
        output_names=['output'],       
        dynamic_axes={                 
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print("量化模型已成功导出为ONNX格式")
except Exception as e:
    print(f"导出ONNX时出错: {str(e)}")