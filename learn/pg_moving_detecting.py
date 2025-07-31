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

import pyautogui as pg
import time

while True:
    mx, my = pg.position()
    print(mx, my)
    time.sleep(0.1)

# 在游戏中检测鼠标位置并无卵用，只有相对位置起作用，所以确实和普通模式有很大区别
#

