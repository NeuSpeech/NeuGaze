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

import keyboard
import time

t_start = time.time()
while True:
    time.sleep(0.1)
    if keyboard.is_pressed('ctrl+space+p'):
        print("pressed")
        break
    else:
        print("not pressed")
    if time.time() - t_start > 10:
        break
