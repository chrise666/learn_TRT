import torch
import torch.nn as nn
import torchvision.models as models


model = models.resnet50(weights=None)
model.load_state_dict(torch.load('C:/Users/1345795/Desktop/model/resnet50-0676ba61.pth'))
model.eval()  # 若存在batchnorm、dropout层则一定要eval()!!!!再export

input_names = ["input"]
output_names = ["output"]

x = torch.randn((1, 3, 224, 224))

torch.onnx.export(model, x, 'C:/Users/1345795/Desktop/model/resnet50.onnx',
                  input_names=input_names, output_names=output_names,
                  dynamic_axes={'input': {0: 'batch'}, 
                                'output': {0: 'batch'}})  # 指定input和output的batch可变

print("Done.!")
