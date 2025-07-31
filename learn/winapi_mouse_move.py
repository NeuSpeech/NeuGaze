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


import win32api
import win32con
import time

def smooth_move_1(dx, dy, duration=2.0, steps=100):
    """
    方案1: 使用win32api.mouse_event实现平滑移动
    dx, dy: 总移动距离
    duration: 总时长(秒)
    steps: 分成多少步执行
    """
    step_x = dx / steps
    step_y = dy / steps
    step_delay = duration / steps
    
    for _ in range(steps):
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(step_x), int(step_y), 0, 0)
        time.sleep(step_delay)


# 使用示例:
def demo_move():
    # 向左移动1000像素
    smooth_move_1(-1000, 0, duration=1, steps=100)
    # time.sleep(1)
    
    # 或者使用方案2
    # smooth_move_2(-1000, 0, duration=2.0, steps=100)


time.sleep(3)

t1=time.time()
demo_move()
t2=time.time()
print(t2-t1)
# 2ms
