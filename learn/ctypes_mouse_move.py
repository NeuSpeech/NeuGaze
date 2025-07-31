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

from ctypes import windll, Structure, c_long

# 加载user32.dll库
user32 = windll.user32

# 定义POINT结构体
class POINT(Structure):
    _fields_ = [("x", c_long), ("y", c_long)]

# 平滑移动鼠标的函数
def smooth_move(start_x, start_y, end_x, end_y, steps=10):
    dx = (end_x - start_x) / steps
    dy = (end_y - start_y) / steps
    for i in range(steps):
        new_x = int(start_x + dx * i)
        new_y = int(start_y + dy * i)
        point = POINT()
        point.x = new_x
        point.y = new_y
        user32.SetCursorPos(point)
        time.sleep(0.01)  # 调整睡眠时间以控制移动速度

# 调用函数
import time
time.sleep(2)
print(time.time())
smooth_move(100, 100, 500, 500)
print(time.time())