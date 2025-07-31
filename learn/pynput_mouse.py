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

import time

from pynput.mouse import Button, Controller

# 鼠标控制器
mouse = Controller()
time.sleep(2)
# 鼠标相对当前位置移动：
print('moving')
mouse.move(250, 250)