import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pandas as pd

# 示例数据，替换为你的数据
file_path = "gradual_elliptical_field_strength.csv"  # 确保路径正确
data = pd.read_csv(file_path)

# 绘制伪彩图
cmap = plt.cm.viridis  # 颜色映射
norm = Normalize(vmin=np.min(data), vmax=np.max(data))  # 归一化

# 创建伪彩图
plt.figure(figsize=(6, 5))
plt.imshow(data, cmap=cmap, norm=norm)
plt.colorbar(label="Value")
plt.title("Pseudocolor Map")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

# 提取 RGB 数据
rgb_data = cmap(norm(data))[:, :, :3]  # 获取 RGB 值 (去掉 alpha 通道)
# rgb_data = (rgb_data * 255).astype(np.uint8)  # 转为 0-255 整数

# 保存 RGB 数据为文件
rgb_df = pd.DataFrame(rgb_data.reshape(-1, 3), columns=["R", "G", "B"])
rgb_df.to_csv("rgb_data.csv", index=False)
print("RGB 数据已导出为 rgb_data.csv")
