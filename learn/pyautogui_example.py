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
import multiprocessing as mp

def key_down_up(key):
    try:
        pg.keyDown(key)
        time.sleep(0.1)  # 按键持续时间
        pg.keyUp(key)
    except Exception as e:
        print(f"Error with key {key}: {e}")

if __name__ == '__main__':
    pool = mp.Pool(processes=8)
    pg.PAUSE = 1
    t0 = time.time()
    time.sleep(2)
    for i in range(1,4):
        # 假设我们按的是数字键，使用正确的按键标识符
        pool.apply_async(key_down_up, args=(str(i % 10),))

    pool.close()  # 关闭pool，不再接受新的任务
    pool.join()   # 等待所有进程完成
    t1 = time.time()
    print(f'total time: {t1 - t0}')
    print(abs(-1))
