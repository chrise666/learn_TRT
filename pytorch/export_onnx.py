import torch
import model # 这里的model是自定义模型的文件名


# 实例化模型
# model = models.resnet50(weights=None)
# model.load_state_dict(torch.load('C:/Users/1345795/Desktop/model/resnet50-0676ba61.pth'))
model = model.SimpleCNN()
model.eval()  # 若存在batchnorm、dropout层则一定要eval()!!!!再export

input_names = ["input"]
output_names = ["output"]

x = torch.randn((1, 3, 32, 32))

torch.onnx.export(model, x, 'onnx/SimpleCNN.onnx',
                  input_names=input_names, output_names=output_names,
                  dynamic_axes={'input': {0: 'batch_size'}, 
                                'output': {0: 'batch_size'}}, # 指定input和output的batch可变
                  opset_version=13,  
                 )  

print("Done.!")
