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

from .resnet import resnet18, resnet34, resnet50
from .mobilenet import mobilenet_v2
from .mobileone import mobileone_s0, mobileone_s1, mobileone_s2, mobileone_s3, mobileone_s4, reparameterize_model
from .scrfd import SCRFD
