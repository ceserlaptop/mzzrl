import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取 RGB 数据
file_path = "rgb_data.csv"  # 替换为你的 RGB 数据文件路径
rgb_data = pd.read_csv(file_path).values  # 读取并转为 NumPy 数组

# 确定图像尺寸 (假设原始数据为10x10)
image_size = (10, 10)  # 替换为实际尺寸
rgb_image = rgb_data.reshape(image_size[0], image_size[1], 3)  # 重新调整形状为 HxWx3

# 绘制图像
plt.figure(figsize=(6, 5))
plt.imshow(rgb_image, origin='lower')  # 显示 RGB 图像
plt.title('RGB Image Representation')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.axis('off')  # 隐藏坐标轴
plt.show()
