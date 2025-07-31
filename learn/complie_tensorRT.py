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

import torch
import torch_tensorrt
import torchvision.models as models

# Initialize model with half precision and sample inputs
model = models.resnet18(pretrained=True).half().eval().to("cuda")
inputs = [torch.randn((1, 3, 224, 224)).to("cuda").half()]

# Enabled precision for TensorRT optimization
enabled_precisions = {torch.half}

# Whether to print verbose logs
debug = True

# Workspace size for TensorRT
workspace_size = 20 << 30

# Maximum number of TRT Engines
# (Lower value allows more graph segmentation)
min_block_size = 7

# Operations to Run in Torch, regardless of converter support
torch_executed_ops = {}

# Build and compile the model with torch.compile, using Torch-TensorRT backend
optimized_model = torch_tensorrt.compile(
    model,
    ir="torch_compile",
    # inputs=inputs,
    enabled_precisions=enabled_precisions,
    debug=debug,
    # workspace_size=workspace_size,
    # min_block_size=min_block_size,
    # torch_executed_ops=torch_executed_ops,
)

# Does not cause recompilation (same batch size as input)
# new_inputs = [torch.randn((1, 3, 224, 224)).half().to("cuda")]
new_inputs = torch.randn((1, 3, 224, 224)).half().to("cuda")
new_outputs = optimized_model(new_inputs)
print(new_outputs)
print(new_outputs.shape,type(new_outputs))

