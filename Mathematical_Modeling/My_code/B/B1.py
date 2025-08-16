import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_excel("data1.xlsx",header = None)

fig = plt.figure()
ax = plt.axes(projection="3d")

x = np.arange(0,4.02,0.02)
y = np.arange(0,5.02,0.02)
X, Y = np.meshgrid(x, y)
Z = np.array(df.values)  # 假设已经将 DataFrame 转换为 NumPy 数组
angle = np.radians(60)   # 将角度转换为弧度

Z1 = Z * np.tan(angle)  # 用 Z 乘以 tan(60°)
surf = ax.plot_surface(X,Y,Z,alpha=0.9, cstride=1, rstride = 1, cmap='rainbow') #Greys
ax.plot_surface(X,Y,Z1,alpha=0.9, cstride=1, rstride = 1, cmap='Blues')

ax.set_xlabel("东西 (海里)",fontsize=12)
ax.set_ylabel("南北 (海里)",fontsize=12)
ax.set_zlabel("探测半径 (米)",fontsize=12)
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, location='left')
plt.title("探测半径")
plt.show()

