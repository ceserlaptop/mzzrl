from alg.utils import generate_field_strength
import pandas as pd

# 设置参数
size = 30
center_x = 15  # 中心点可以自定义
center_y = 15  # 中心点可以自定义
max_strength = 100  # 最大场强值可以自定义
radius = 10      # 影响半径可以自定义

# 生成场强数据
field_data = generate_field_strength(size, center_x, center_y, max_strength, radius)

# 将数据转换为 DataFrame
df = pd.DataFrame(field_data)

# 将 DataFrame 保存为 CSV 文件
df.to_csv("field_strength_data.csv", index=False, header=True)
