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

from my_model_arch.my_l2cs.pipeline import IntegratedRegressionWithKeys9Num,ObserverWithSectorWheel
import torch
from pathlib import Path
from multiprocessing import freeze_support
import threading
import os
import warnings

# 关闭特定警告
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype() is deprecated.*")


if __name__ == '__main__':
    freeze_support()
    CWD=Path(__file__).parent

    pipeline = IntegratedRegressionWithKeys9Num(
        regression_model='lassocv',
        weights=CWD / 'models' / 'L2CSNet_gaze360.pkl',
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        func='random',
        render_in_eval=False,
        kalman_filter_std_measurement=2,
        num_points=16,
        images_freq=25,
        each_point_wait_time=1000,
        every_point_has_n_images=25,
    )
    observer=ObserverWithSectorWheel(pipeline,radius=800)
    root_dir = 'calibration'
    all_dirs = os.listdir(root_dir)
    useful_dirs = []
    for _dir in all_dirs:
        images = os.listdir(os.path.join(root_dir, _dir, 'images'))
        if len(images) == 400:
            useful_dirs.append(os.path.join(root_dir, _dir, 'train_data.jsonl'))
    pipeline.load_model('model_weights/20240924_092902/model.pkl')
    # pipeline.train_lot_data(
    #     jsonl_path_list=useful_dirs,
    #     model_save_path='models/eye_tracking.pkl'
    #
    # pipeline.demo(0,start_with_calibration=False)
    # threading.Thread(target=pipeline.run_sector_wheel,args=(500,)).start()
    threading.Thread(target=pipeline.demo,args=(0,False)).start()
    threading.Thread(target=observer.run_sector_wheel).start()

    # pipeline.demo(0,start_with_calibration=True)
