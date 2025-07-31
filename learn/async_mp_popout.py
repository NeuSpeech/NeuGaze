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

# import os
# import sys
# import multiprocessing
# import time
#
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from my_model_arch.my_l2cs.vis import popout_fading_window
#
# class A:
#     def __init__(self):
#         pass
#
#     def call(self):
#         processes = []
#         for i in range(5):
#             # 创建一个 Process 对象，并指定 target 函数和参数
#             p = multiprocessing.Process(target=popout_fading_window, args=("NumLock is ON", 1))
#             processes.append(p)
#             p.start()  # 启动进程
#             time.sleep(0.1)  # 稍微延迟以避免同时启动所有进程
#
#         # 等待所有进程完成
#         for p in processes:
#             p.join()
#
# a = A()
# a.call()

import os
import sys
import multiprocessing
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from my_model_arch.my_l2cs.vis import popout_fading_window

class A:
    def __init__(self):
        pass

    def call(self):
        processes = []
        for i in range(5):
            # 创建一个 Process 对象，并指定 target 函数和参数
            p = multiprocessing.Process(target=popout_fading_window, args=("NumLock is ON", 1))
            processes.append(p)
            p.start()  # 启动进程
            time.sleep(0.1)  # 稍微延迟以避免同时启动所有进程

        # 等待所有进程完成
        for p in processes:
            p.join()

if __name__ == '__main__':
    a = A()
    a.call()



