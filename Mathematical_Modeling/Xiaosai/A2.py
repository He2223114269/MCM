import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
# 读取表1中的数据

data_1 = pd.read_excel('1.xlsx', header=None)

# 将数据存储在向量中
meanKongXiLiuSu = data_1.iloc[0, 0]    # 平均孔隙流速
groundwaterSeepage = data_1.iloc[0, 1]    # 地下水渗流流速
permeability = data_1.iloc[0, 2]    # 渗透系数
dispersionCoefficient = data_1.iloc[0, 3]    # 弥散系数
dryDensity = data_1.iloc[0, 4]  # 含水层样品的干密度
porosity = data_1.iloc[0, 5]    # 孔隙度

L = 1  # 柱子长度，单位m
t = 10  # 试验时间，单位d

# 计算有效孔隙度和孔径
effectivePorosity = dryDensity * porosity  # 有效孔隙度
poreSize = (permeability / effectivePorosity)**(1/2)  # 孔径

# 计算阻带数
R = 1/4 * (meanKongXiLiuSu / groundwaterSeepage + 1)**2 + 2 * (dispersionCoefficient / (poreSize**2 * groundwaterSeepage)) + 1/2 * (meanKongXiLiuSu / groundwaterSeepage - 1)**2

# 计算扩散时间
tau_D = L**2 / (4 * dispersionCoefficient)

# 计算有效对流时间
tau_c = L / meanKongXiLiuSu

# 计算Peclet数
Pe = meanKongXiLiuSu * poreSize / dispersionCoefficient  # Peclet数是描述物质在流体中传递过程中对流传递和扩散传递比例的一个无量纲数

# 输出结果
print(f'有效孔隙度为 {effectivePorosity}')
print(f'孔径大小为 {poreSize:.4f} cm')
print(f'延缓因子为 {R:.4f}')
print(f'扩散时间为 {tau_D:.4f} d')
print(f'有效迁流时间为 {tau_c:.4f} d')
print(f'Peclet数为 {Pe:.4f}')

# 读取表2中的数据

# 读取表2中的数据
data = pd.read_excel('2.xlsx')

# 提取液相和固相浓度数据
t = data.iloc[:, 0]
S1_liquid = data.iloc[:, 1]
S1_solid = data.iloc[:, 2]
S2_liquid = data.iloc[:, 3]
S2_solid = data.iloc[:, 4]
S3_liquid = data.iloc[:, 5]
S3_solid = data.iloc[:, 6]
S4_liquid = data.iloc[:, 7]
S4_solid = data.iloc[:, 8]

# 计算各样品液相和固相的平均浓度
S1_liquid_avg = np.mean(S1_liquid)
S1_solid_avg = np.mean(S1_solid)
S2_liquid_avg = np.mean(S2_liquid)
S2_solid_avg = np.mean(S2_solid)
S3_liquid_avg = np.mean(S3_liquid)
S3_solid_avg = np.mean(S3_solid)
S4_liquid_avg = np.mean(S4_liquid)
S4_solid_avg = np.mean(S4_solid)

# 绘制液相浓度和时间的关系图
plt.figure()
plt.subplot(2,1,1)
plt.plot(t, S1_liquid, t, S2_liquid, t, S3_liquid, t, S4_liquid,)
# plt.plot(t, S1_liquid,color='yellow' )
# plt.plot(t, S2_liquid,color='green')
# plt.plot(t,S3_liquid,color='red')
# plt.plot(t, S4_liquid,color = "black")
# plt.xlabel('时间 (h)')
plt.title("液体\固体浓度与时间的关系")
plt.ylabel('液浓度 (mg/L)')
plt.legend(['样品1', '样品2', '样品3', '样品4'])

# 绘制固相浓度和时间的关系图
# plt.figure()
plt.subplot(2,1,2)
plt.plot(t, S1_solid, t, S2_solid, t, S3_solid, t, S4_solid,)
# plt.plot(t, S1_solid,color='yellow' )
# plt.plot(t, S2_solid,color='green')
# plt.plot(t,S3_solid,color='red')
# plt.plot(t, S4_solid,color = "black")
plt.xlabel('时间 (h)')
plt.ylabel('固浓度 (mg/Kg)')
plt.legend(['样品1', '样品2', '样品3', '样品4'])
plt.show()
# 计算各样品液相和固相的分配系数,分配系数Kd是描述有机污染物在水和固体之间分配平衡的重要参数
# 其值越大，表明固体对有机污染物的吸附能力越强，即有机污染物更容易在水体中向固体相转移。
# 因此，可以通过比较不同样品的分配系数，评估固体吸附能力的差异，从而判断不同的地下水和河流环境对于有机污染物的迁移和转化过程的影响。
Kd1 = S1_solid_avg / S1_liquid_avg
Kd2 = S2_solid_avg / S2_liquid_avg
Kd3 = S3_solid_avg / S3_liquid_avg
Kd4 = S4_solid_avg / S4_liquid_avg

Kd_S1 = S1_solid_avg / S1_liquid_avg
Kd_S2 = S2_solid_avg / S2_liquid_avg
Kd_S3 = S3_solid_avg / S3_liquid_avg
Kd_S4 = S4_solid_avg / S4_liquid_avg

qmax_S1 = S1_solid_avg * (1 - 1 / Kd_S1)
qmax_S2 = S2_solid_avg * (1 - 1 / Kd_S2)
qmax_S3 = S3_solid_avg * (1 - 1 / Kd_S3)
qmax_S4 = S4_solid_avg * (1 - 1 / Kd_S4)

# 输出各样品的液相和固相平均浓度，以及液相和固相的分配系数
print(f'样品1液相平均浓度：{S1_liquid_avg} mg/Kg')
print(f'样品1固相平均浓度：{S1_solid_avg} mg/Kg')
print(f'样品1分配系数：{Kd1}, 最大吸附量：{qmax_S1} mg/kg')

print(f'样品2液相平均浓度：{S2_liquid_avg} mg/L')
print(f'样品2固相平均浓度：{S2_solid_avg} mg/Kg')
print(f'样品2分配系数：{Kd2}, 最大吸附量：{qmax_S2} mg/kg')

print(f'样品3液相平均浓度：{S3_liquid_avg} mg/L')
print(f'样品3固相平均浓度：{S3_solid_avg} mg/Kg')
print(f'样品3分配系数：{Kd3}, 最大吸附量：{qmax_S3} mg/kg')

print(f'样品4液相平均浓度：{S4_liquid_avg} mg/L')
print(f'样品4固相平均浓度：{S4_solid_avg} mg/Kg')
print(f'样品4分配系数：{Kd4}, 最大吸附量：{qmax_S4} mg/kg')

# 读取表3中的数据
import pandas as pd
data = pd.read_excel('3.xlsx')

# 提取液相和固相浓度数据
initial_conc = data.iloc[:, 0]
S1_liquid_conc = data.iloc[:, 1]
S1_solid_conc = data.iloc[:, 2]
S2_liquid_conc = data.iloc[:, 3]
S2_solid_conc = data.iloc[:, 4]
S3_liquid_conc = data.iloc[:, 5]
S3_solid_conc = data.iloc[:, 6]
S4_liquid_conc = data.iloc[:, 7]
S4_solid_conc = data.iloc[:, 8]

# 计算各样品液相和固相的平均浓度
S1_liquid_mean = S1_liquid_conc.mean()
S1_solid_mean = S1_solid_conc.mean()
S2_liquid_mean = S2_liquid_conc.mean()
S2_solid_mean = S2_solid_conc.mean()
S3_liquid_mean = S3_liquid_conc.mean()
S3_solid_mean = S3_solid_conc.mean()
S4_liquid_mean = S4_liquid_conc.mean()
S4_solid_mean = S4_solid_conc.mean()

import matplotlib.pyplot as plt
# 绘制等温线
plt.plot(S1_solid_conc, S1_liquid_conc, '--')
plt.plot(S2_solid_conc, S2_liquid_conc, '--')
plt.plot(S3_solid_conc, S3_liquid_conc, '--')
plt.plot(S4_solid_conc, S4_liquid_conc, '--')
plt.title('等温线')
plt.xlabel('固相浓度 (mg/kg)')
plt.ylabel('液相浓度 (mg/L)')
plt.legend(['S1', 'S2', 'S3', 'S4'])
plt.show()
# 计算各样品的等温吸附系数和最大吸附量
Kd_S1 = S1_solid_mean / S1_liquid_mean
Kd_S2 = S2_solid_mean / S2_liquid_mean
Kd_S3 = S3_solid_mean / S3_liquid_mean
Kd_S4 = S4_solid_mean / S4_liquid_mean
qmax_S1 = S1_solid_conc.iloc[-1] * (1 - 1 / Kd_S1)
qmax_S2 = S2_solid_conc.iloc[-1] * (1 - 1 / Kd_S2)
qmax_S3 = S3_solid_conc.iloc[-1] * (1 - 1 / Kd_S3)
qmax_S4 = S4_solid_conc.iloc[-1] * (1 - 1 / Kd_S4)

# 输出结果
print(f'样品S1: 等温吸附系数 = {Kd_S1:.4f} L/kg, 最大吸附量 = {qmax_S1:.4f} mg/kg')
print(f'样品S2: 等温吸附系数 = {Kd_S2:.4f} L/kg, 最大吸附量 = {qmax_S2:.4f} mg/kg')
print(f'样品S3: 等温吸附系数 = {Kd_S3:.4f} L/kg, 最大吸附量 = {qmax_S3:.4f} mg/kg')
print(f'样品S4: 等温吸附系数 = {Kd_S4:.4f} L/kg, 最大吸附量 = {qmax_S4:.4f} mg/kg')