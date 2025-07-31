# =============================================================================
# NeuSpeech Institute, NeuGaze Project
# Copyright (c) 2024 Yiqian Yang
#
# This code is part of the NeuGaze project developed at NeuSpeech Institute.
# Author: Yiqian Yang
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 
# International License. To view a copy of this license, visit:
# http://creativecommons.org/licenses/by-nc/4.0/
# =============================================================================

from my_model_arch.my_l2cs.model import L2CS
import jsonlines
import math
import time
import copy
import torchvision
import torch
import torch_tensorrt


def getArch(arch, bins):
    # Base network structure
    if arch == 'ResNet18':
        model = L2CS(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], bins)
    elif arch == 'ResNet34':
        model = L2CS(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3], bins)
    elif arch == 'ResNet101':
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], bins)
    elif arch == 'ResNet152':
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 8, 36, 3], bins)
    else:
        if arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                  'The default value of ResNet50 will be used instead!')
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], bins)
    return model


arch='ResNet50'
weights='models/L2CSNet_gaze360.pkl'
device='cuda'
model = getArch(arch, 90)
model.load_state_dict(torch.load(weights, map_location=device))
model.half()
model.eval()
model.cuda()
# torch_tensorrt.save(model, "models/L2CSNet_gaze360_fp16.ts",output_format="torchscript", inputs=inputs)
x = torch.randn((1, 3, 448, 448)).half().cuda() # define what the inputs to the model will look like

# optimized_model = torch.compile(model, backend="tensorrt")
# optimized_model(x) # compiled on first run
#
# optimized_model(x) # this will be fast!
inputs=[x]
trt_gm = torch_tensorrt.compile(model, ir="dynamo", inputs=inputs)
torch_tensorrt.save(trt_gm, "models/L2CSNet_gaze360_fp16.ep", inputs=inputs) # PyTorch only supports Python runtime for an ExportedProgram. For C++ deployment, use a TorchScript file
torch_tensorrt.save(trt_gm, "models/L2CSNet_gaze360_fp16.ts", output_format="torchscript", inputs=inputs)


data_path='models/data.pt'
data=torch.load(data_path)
input_data=data['img']
model = torch.export.load("models/L2CSNet_gaze360_fp16.ep").module()
out=model(input_data)
print(out)
# 成功