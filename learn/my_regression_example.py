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

from my_model_arch.my_l2cs import RegressionPipeline
import torch
from pathlib import Path


CWD=Path(__file__).parent

pipeline = RegressionPipeline(
    weights=CWD / 'models' / 'L2CSNet_gaze360.pkl',
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    func='sequence',
    num_points=16,
    images_freq=25,
    each_point_wait_time=1000,
    every_point_has_n_images=25,
)
# FPS 11-12
# FPS:11.770558655031914
# model time:0.011383295059204102s
#
# predict_gaze time:0.015593290328979492s
#
# t_eval_box-t_eval_start:0.07923746109008789s
# t_eval_pred-t_eval_box:0.0010466575622558594s
# t_eval_smooth-t_eval_pred:0.0s
# t_eval_render-t_eval_smooth:0.0s
# t_eval_show-t_eval_render:0.0021643638610839844s
pipeline.load_model('model_weights/20240906_094821/model.pkl')
pipeline.demo(0,start_with_calibration=False)