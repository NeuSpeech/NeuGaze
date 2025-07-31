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

import numpy as np
from threading import Thread
import threading
lock = threading.Lock()
result=None

def return_np(i):
    with lock:
        global result
        result=np.arange(i+1)
        return result


for i in range(5):
    Thread(target=return_np, args=(i,)).start()
    print(i,type(result),result)