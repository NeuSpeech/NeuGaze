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

from pynput.keyboard import Controller, Key
import time

# 创建键盘控制器对象
keyboard = Controller()

# 模拟按下 'a' 键
keyboard.press('w')

# 保持 'a' 键按下的状态一段时间（例如，5秒）
time.sleep(5)

# 模拟释放 'a' 键
keyboard.release('w')
