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

from my_model_arch.my_l2cs import RegressionMediaPipeline
import torch
from pathlib import Path


CWD=Path(__file__).parent

pipeline = RegressionMediaPipeline(
    regression_model='lassocv',
    weights=CWD / 'models' / 'L2CSNet_gaze360.pkl',
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    func='sequence',
    kalman_filter_std_measurement=0.5,
    num_points=16,
    images_freq=25,
    each_point_wait_time=1000,
    every_point_has_n_images=25,
)

pipeline.load_model('model_weights/20240903_120859/model.pkl')
pipeline.demo(0,start_with_calibration=False)
