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
import threading

def press_key(key_code, scancode):
    """模拟按键按下，使用扫描码"""
    win32api.keybd_event(key_code, scancode, 0, 0)

def release_key(key_code, scancode):
    """模拟按键释放，使用扫描码"""
    win32api.keybd_event(key_code, scancode, win32con.KEYEVENTF_KEYUP, 0)

# 定义按键的虚拟键码和扫描码
SHIFT_KEY = win32con.VK_LSHIFT
SHIFT_SCANCODE = 0x2A  # 左Shift的扫描码
W_KEY = ord('E')
W_SCANCODE = 0x11  # W键的扫描码

def hold_keys(duration):
    """同时按住W和Shift键指定时间"""
    try:
        # 先按Shift，等待一小段时间再按W
        # press_key(SHIFT_KEY, SHIFT_SCANCODE)
        # time.sleep(0.05)
        press_key(W_KEY, W_SCANCODE)
        
        time.sleep(duration)
        
        # 先释放W，再释放Shift
        release_key(W_KEY, W_SCANCODE)
        # time.sleep(0.05)
        # release_key(SHIFT_KEY, SHIFT_SCANCODE)
    except:
        # 确保按键被释放
        release_key(W_KEY, W_SCANCODE)
        release_key(SHIFT_KEY, SHIFT_SCANCODE)

def continuous_pressing(duration=5, interval=0.12):
    """连续按住W和Shift键指定时间，可以设置间隔"""
    end_time = time.time() + duration
    while time.time() < end_time:
        press_key(SHIFT_KEY, SHIFT_SCANCODE)
        time.sleep(0.05)
        press_key(W_KEY, W_SCANCODE)
        
        time.sleep(interval)
        
        release_key(W_KEY, W_SCANCODE)
        time.sleep(0.05)
        release_key(SHIFT_KEY, SHIFT_SCANCODE)
        
        time.sleep(0.02)

if __name__ == "__main__":
    print("程序将在2秒后开始执行...")
    time.sleep(2)
    
    # 测试方式1：持续按住
    hold_keys(5)
    # hold_keys(0.1)
    # hold_keys(0.1)
    # hold_keys(0.1)
    # hold_keys(0.1)
    # for i in range(200):
    #     thread = threading.Thread(target=hold_keys, args=(0.2,))
    #     thread.start()
    #     time.sleep(0.02)

    # 或者测试方式2：间歇性按住
    # continuous_pressing(5, 0.12)