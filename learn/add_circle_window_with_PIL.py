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

from PIL import Image, ImageDraw
import io

# 创建一个透明的图像
size = (400, 400)
background = Image.new('RGBA', size, (255, 255, 255, 0))

# 创建一个圆形
circle_size = 200
circle_center = (size[0] // 2, size[1] // 2)
circle_draw = ImageDraw.Draw(background)
circle_draw.ellipse(
    [circle_center[0] - circle_size // 2, circle_center[1] - circle_size // 2,
     circle_center[0] + circle_size // 2, circle_center[1] + circle_size // 2],
    fill=(255, 255, 255, 128)  # 半透明的白色
)

# 保存图像到内存中的BytesIO对象
img_byte_arr = io.BytesIO()
background.save(img_byte_arr, format='PNG')
img_byte_arr = img_byte_arr.getvalue()

# 显示图像（在这里，我将使用base64编码来展示图像，因为我们的环境不支持直接显示图像）
import base64
img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')
html = f'<img src="data:image/png;base64,{img_base64}">'
html
