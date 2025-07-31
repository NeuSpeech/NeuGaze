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

from my_model_arch.my_l2cs.pipeline import RealAction
# import torch
from pathlib import Path
# from multiprocessing import freeze_support
# import threading
# import os
import warnings
import yaml
warnings.filterwarnings("ignore")
# 关闭特定警告
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype() is deprecated.*")
# import torch_tensorrt
if __name__ == '__main__':
    # freeze_support()
    CWD = Path(__file__).parent
    with open(CWD / 'all.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    pipeline = RealAction(**config['real_action_config'],
                          gaze_config=config['gaze_config'],
                          mouse_control_config=config['mouse_control_config'],
                          wheel_config=config['wheel_config'],
                          configuration=config['key_config'],
                          head_angles_center=config['head_angles_center'],
                          head_angles_scale=config['head_angles_scale'],
                          expression_evaluator_config=config['expression_evaluator_config'],
                          **config['integrated_config']
                          )

    # pipeline = RealAction(
    #     regression_model='lassocv',
    #     # weights=CWD / 'models' / 'L2CSNet_gaze360.pkl',
    #     # weights=CWD / 'models' / 'L2CSNet_gaze360_fp16.ts',
    #     # weights=CWD / 'models' / 'L2CSNet_gaze360_fp16.ep',
    #     weights=CWD / 'gaze-estimation' / 'weights' / 'mobileone_s0_fp16.ep',
    #     # arch='mobileone_s0',
    #     # weights=CWD / 'gaze-estimation' / 'weights' / 'mobileone_s0.pt',
    #     device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    #     func='random',
    #     render_in_eval=False,
    #     # gaze_bias=(400,200),
    #     gaze_bias=(800,200),
    #     # gaze_bias=(400,-100),
    #     kalman_filter_std_measurement=4,
    #     num_points=4,
    #     images_freq=25,
    #     each_point_wait_time=1000,
    #     every_point_has_n_images=25,
    #     sys_mode='game',
    # )
    # observer = ObserverWithSectorWheel(pipeline, radius=1000)
    # root_dir = 'calibration'
    # all_dirs = os.listdir(root_dir)
    # useful_dirs = []
    # for _dir in all_dirs:
    #     images = os.listdir(os.path.join(root_dir, _dir, 'images'))
    #     if len(images) == 400:
    #         useful_dirs.append(os.path.join(root_dir, _dir, 'train_data.jsonl'))
    # 这个是白天的模型，因为光照条件不一样。
    # pipeline.load_model('model_weights/20240924_092902/model.pkl')
    # 晚上光照更均匀，所以用白天的模型会偏移。
    # pipeline.train_lot_data(
    #     jsonl_path_list=useful_dirs,
    #     model_save_path='models/eye_tracking.pkl')
    #
    # pipeline.demo(0,start_with_calibration=False)
    # threading.Thread(target=pipeline.run_sector_wheel,args=(500,)).start()
    # threading.Thread(target=pipeline.demo, args=(0, False)).start()

    pipeline.demo()
