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

from my_model_arch.my_l2cs.pipeline import OnlyMediaPipe
import torch
from pathlib import Path
import os


CWD=Path(__file__).parent

pipeline = OnlyMediaPipe(
    regression_model='lassocv',
    weights=CWD / 'models' / 'L2CSNet_gaze360.pkl',
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    func='random',
    kalman_filter_std_measurement=2,
    num_points=16,
    images_freq=25,
    each_point_wait_time=1000,
    every_point_has_n_images=25,
    render_in_eval=True,


)

root_dir = 'calibration'
all_dirs = os.listdir(root_dir)
# select with time
all_dirs=[x for x in all_dirs if int(x.split('_')[0])>=20241125][2:]
useful_dirs = []
for _dir in all_dirs:
    images = os.listdir(os.path.join(root_dir, _dir, 'images'))
    if len(images) == 400:
        useful_dirs.append(os.path.join(root_dir, _dir, 'train_data.jsonl'))
# pipeline.train_lot_data(
#     jsonl_path_list=useful_dirs,
#     model_save_path='models/eye_tracking.pkl')
# pipeline.load_model('model_weights/20240918_224556/model.pkl')
pipeline.regression_model.load_state_dict(torch.load('model_weights/only_mediapipe/best_model.pth'))
pipeline.regression_model.to(pipeline.device)
pipeline.demo(0,start_with_calibration=False)
