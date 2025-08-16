import numpy as np
import matplotlib.pyplot as plt

# 给定参数
L = 1000  # 模拟距离，单位为米
T = 100  # 模拟时间，单位为天
dx = 10  # 空间步长，单位为米
dt = 0.1  # 时间步长，单位为天
v = 5.01 / 100  # 地下水渗流流速，单位为米每天
D = 0.38 / 100 / 60 * 1440  # 弥散系数，单位为米平方每天
k = 0.1  # 吸附系数，单位为1/天
C0 = 1  # 初始浓度，单位为毫克/升

# 计算网格数
nx = int(L / dx) + 1    # 空间步长
nt = int(T / dt) + 1    # 时间步长

# 初始化浓度矩阵
C = np.zeros((nt, nx))
C[0, :] = C0

# 迭代求解


for i in range(1, nt):
    # 计算对流项
    adv = -v * (C[i-1, :] - C[i-1, :]) / dx * dt
    # 计算扩散项
    dif = D * (C[i-1, :] - 2 * C[i-1, :] + C[i-1, :]) / dx**2 * dt
    # 计算吸附项
    ads = -k * C[i-1, :] * dt
    # 更新浓度
    C[i, :] = C[i-1, :] + adv + dif + ads

#对时间空间求值
# for i in range(1, nt):
#     for j in range(1, nx):
#         # 计算对流项
#         adv = -v * (C[i-1, j] - C[i-1, j]) / dx * dt
#         # 计算扩散项
#         dif = D * (C[i-1, j] - 2 * C[i-1, j] + C[i-1, j]) / dx**2 * dt
#         # 计算吸附项
#         ads = -k * C[i-1, j] * dt
#         # 更新浓度
#         C[i, j] = C[i-1, j] + adv + dif + ads

# 绘制浓度随时间和距离的变化图像
x = np.linspace(0, L, nx)
t = np.linspace(0, T, nt)
X, T = np.meshgrid(x, t)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, C, cmap='coolwarm')
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Time (d)')
ax.set_zlabel('Concentration (mg/L)')
ax.set_title('Liquid Concentration')
plt.show()