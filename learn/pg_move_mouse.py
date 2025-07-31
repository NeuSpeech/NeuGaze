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

import pydirectinput
import time
pydirectinput._handlePause(0.01)
time.sleep(2)  # 等待一秒钟
# pg.moveRel(500, 0)  # 向右移动鼠标100像素
# time.sleep(1)  # 等待一秒钟
# pg.moveRel(-500, 0)  # 向左移动鼠标100像素，回到原位
# time.sleep(1)  # 等待一秒钟
print(time.time())
pydirectinput.moveRel(xOffset=-1000,yOffset=0,duration=2.0,relative=True)
# for i in range(5):
#     pydirectinput.moveRel(xOffset=-100,yOffset=0,duration=2,relative=False)
print(time.time())

# 1728899887.674024
# 1728899887.78229

# 这个函数移动鼠标的确是需要0.11s左右的时间，所以之后使用的时候要防止执行操作序列累积。每次移动0.15s扫描一次比较合适。