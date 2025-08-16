import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 数据准备
X = np.array([1,2,3,4,5,6,8,9,10])
# Y = np.array([1.3, 1.36, 1.41, 1.51, 1.64, 1.64, 1.5, 1.56, 1.6, 1.63, ])# 每一农村劳动力负担系数/人
# Y = np.array([2.985893417,3.044124817,3.037618942,2.945852535,2.826456895,2.701214575,2.232150227,2.189968478,2.167940228,2.173708552,2.169715815,2.157810627,2.151906481,2.109799224,2.065689271,])#城乡居民收入倍差/比值
Y = np.array([43400,
76767,
215843,
154233,
177764,
251214,

275827,
304091,
584696,
])#城镇化率/%

# Y = np.array([1.53,1.89,1.81,1.53,1.45,1.45,1.39,1.51,1.54,1.51])# 每一城镇就业者负担系数/人


new_x = np.array(7)

import scipy.interpolate as spi

ipo1 = spi.splrep(X, Y, k=1)
y1 = spi.splev(new_x, ipo1)

ipo2 = spi.splrep(X, Y, k=2)
y2 = spi.splev(new_x, ipo2)

ipo3 = spi.splrep(X, Y, k=3)
y3 = spi.splev(new_x, ipo3)

ipo4 = spi.splrep(X, Y, k=4)
y4 = spi.splev(new_x, ipo4)
print(f"y1: {y1}\n y2: {y2}\n y3: {y3}\n y4:{y4}")
fig, axes = plt.subplots(2, 2, figsize=(10, 12))
ax1, ax2 = axes[0, :]  # 获取第一行的两个Axes
ax3, ax4 = axes[1, :]  # 获取第二行的两个Axes

# 画图

ax1.plot(X, Y, '-o', label='样本点')
ax1.plot(new_x, y1, '-o', label='插值点')
ax1.set_ylim(Y.min() - 1, Y.max() + 1)
ax1.set_ylabel('指数')
ax1.set_title('线性插值')
ax1.legend()

ax2.plot(X, Y, '-o', label='样本点')
ax2.plot(new_x, y2, '-o', label='插值点')
ax2.set_ylim(Y.min() - 1, Y.max() + 1)
ax2.set_ylabel('指数')
ax2.set_title('二次插值')
ax2.legend()

ax3.plot(X, Y, '-o', label='样本点')
ax3.plot(new_x, y3, '-o', label='插值点')
ax3.set_ylim(Y.min() - 1, Y.max() + 1)
ax3.set_ylabel('指数')
ax3.set_title('三次插值')
ax3.legend()

ax4.plot(X, Y, '-o', label='样本点')
ax4.plot(new_x, y4, '-o', label='插值点')
ax4.set_ylim(Y.min() - 1, Y.max() + 1)
ax4.set_ylabel('指数')
ax4.set_title('四次插值')
ax4.legend()

plt.show()
