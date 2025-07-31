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

from my_model_arch.my_l2cs import Pipeline
import torch
from pathlib import Path


CWD=Path(__file__).parent

pipeline = Pipeline(
    model_weights_path=CWD / 'models' / 'L2CSNet_gaze360.pkl',
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    func='random',
    num_points=50,
    model_class='MyBinsL2CS',
    train_weight_names=['fc_x_gaze.weights','fc_y_gaze.weights','fc_x_gaze.bias','fc_y_gaze.bias'],
    batch_size=16,
    num_epochs=100,
    images_freq=5,
    each_point_wait_time=1000,
    every_point_has_n_images=5,
)

pipeline.demo()