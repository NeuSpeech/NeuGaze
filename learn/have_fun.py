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

import turtle
import random

def draw_rainbow_spiral():
    # 设置窗口
    screen = turtle.Screen()
    screen.bgcolor("black")
    screen.title("放松一下 - 彩虹螺旋")
    
    # 创建画笔
    t = turtle.Turtle()
    t.speed(0)  # 最快速度
    t.width(2)
    
    # 彩虹颜色
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
    
    # 画螺旋
    for i in range(300):
        t.pencolor(colors[i % 6])
        t.forward(i)
        t.left(59)
    
    # 点击关闭窗口
    screen.exitonclick()

if __name__ == "__main__":
    draw_rainbow_spiral()
