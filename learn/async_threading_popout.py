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

# this file studies how to pop out a window while keeping the main thread running
import os,sys,threading,time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from my_model_arch.my_l2cs.vis import popout_fading_window

def fun(i):
    while True:
        print(i,time.time())
        time.sleep(1)

class A:
    def __init__(self):
        pass
    def call(self):
        for i in range(5):
            threading.Thread(target=fun,args=(i,)).start()
            time.sleep(0.1)


a=A()
a.call()

