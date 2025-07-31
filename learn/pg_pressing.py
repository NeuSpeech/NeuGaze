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

import pyautogui as pg
import time
import threading

kd={'w':True}

def real_press(*args):
    pg.press(*args)
    # pg.keyUp(args[0])

def press(key):
    threading.Thread(target=real_press,args=(key,10,0.02,None,0.2)).start()

time.sleep(2)
# lock=threading.Lock()
t0=time.time()
# pg.keyDown('shift')
pg.keyDown('w')
pg.keyDown('w')
pg.keyDown('w')
pg.keyDown('w')
pg.keyDown('w')
pg.keyDown('w')
# while True:
# for i in range(21):
    # with lock:
    # threading.Thread(target=press,args=('shift',)).start()
    # pg.keyDown('shift')
    # if i==19:
    #     kd['w']=False
    # print(pg.hold())
    # threading.Thread(target=pg.press,args=('w',30,0.02,None,1.0)).start()
        # threading.Thread(target=pg.press,args=('w',1,0.0,None,1.0)).start()
    # time.sleep(0.12)
# pg.keyUp('w')
t1=time.time()
print(t1-t0)
# 这个即使没有设置延迟，仍然会卡，说明可能方向键需要特殊处理

# 如果设置了多线程，用多线程来press，那就可以有一个连续的效果。

# 所以其实就是下一次按压的时间要开始在上一次按压结束之前

# 0.02刚刚好可以不抖，但是0.033就会抖，也就是大概FPS拉满时的一个时长会抖，所以我用头部控制是抖动的

# 这个解决办法是设置_PAUSE为0.4，拖到下一次press开始就好了。