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
from win32gui import GetCursorInfo



def is_cursor_visible():
    """检查鼠标光标是否可见"""
    flags, hcursor, (x, y) = GetCursorInfo()
    return hcursor != 0

time.sleep(2)
for i in range(10):
    print(is_cursor_visible())
    time.sleep(0.2)