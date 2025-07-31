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
import torchvision.models as models
import pnnx
import sys
sys.path.append(r"C:\use\research\EEG\eye\real_time_eye_tracker\gaze-estimation")

from utils.helpers import get_model, draw_bbox_gaze
print(get_model("mobileone_s0", 90, pretrained=True, inference_mode=True))
device = torch.device("cpu")
# model = get_model("mobileone_s0", bins, pretrained=True, inference_mode=inference_mode)
weights_path = r"C:\use\research\EEG\eye\real_time_eye_tracker\gaze-estimation\output\gaze360_mobileone_s0_1748661048\mobilenet_s0_224_fused.pt"
gaze_detector = get_model("mobileone_s0", 90, pretrained=False, inference_mode=True)
state_dict = torch.load(weights_path, map_location=device)
gaze_detector.load_state_dict(state_dict)

# Use FP16 input for export
x = torch.rand(1, 3, 224, 224)

# Export with FP16 model and input
opt_model = pnnx.export(gaze_detector, "mobileone_s0_224_fp16_pnnx.pt", x)
print(opt_model)