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

import asyncio
import time


async def f1():
    for i in range(3):
        print(i)
        await asyncio.sleep(1)


async def f2():
    for i in range(3):
        print(f"a{i}")
        await asyncio.sleep(1)


# 这样是同步的逻辑，不会同时执行两个异步函数
asyncio.run(f1())
asyncio.run(f2())

# 直接这样就没用的。
# asyncio.gather(f1(), f2())

# 这个方式是可以的
# async def f3():
#     await asyncio.gather(f1(), f2())
#
# asyncio.run(f3())

# async def f3():
#     t1=asyncio.create_task(f1())
#     t2=asyncio.create_task(f2())
#     await t1
#     await t2
#
# asyncio.run(f3())
