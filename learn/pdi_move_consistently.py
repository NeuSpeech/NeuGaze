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

# import numpy as np
# import threading
# import time
# import pydirectinput
# # pydirectinput._handlePause(0.01)
# pydirectinput.FAILSAFE = False
# time.sleep(2)
# # t0=time.time()
# # for i in range(100):
# #     threading.Thread(target=pydirectinput.moveRel,kwargs={
# #         'xOffset': -10,
# #         'yOffset': 0,
# #         'duration': 2.0,
# #         'relative': True
# #     }).start()
# #     time.sleep(0.01)
# # t1=time.time()
# # print(t1-t0)

# for i in range(100):
#     pydirectinput.moveRel(xOffset=-10,yOffset=0,duration=2.0,relative=True)
#     time.sleep(0.01)


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

def smooth_move_2(dx, dy, duration=2.0, steps=100):
    """
    方案2: 使用SetCursorPos实现平滑移动
    这种方式可能对某些游戏不起作用，因为有些游戏会忽略SetCursorPos
    """
    start_x, start_y = win32api.GetCursorPos()
    step_delay = duration / steps
    
    for i in range(steps):
        progress = (i + 1) / steps
        current_x = int(start_x + dx * progress)
        current_y = int(start_y + dy * progress)
        win32api.SetCursorPos((current_x, current_y))
        time.sleep(step_delay)

# 使用示例:
def demo_move():
    # 向左移动1000像素
    smooth_move_1(-100, 0, duration=0, steps=1)
    # time.sleep(1)
    
    # 或者使用方案2
    # smooth_move_2(-1000, 0, duration=2.0, steps=100)


time.sleep(2)

t1=time.time()
demo_move()
t2=time.time()
print(t2-t1)

# threading.Thread(target=smooth_move_1,kwargs={
#     'xOffset': -10,
#     'yOffset': 0,
#     'duration': 2.0,
#     'relative': True
# }).start()
# for i in range(100):
#     demo_move()
#     time.sleep(0.01)