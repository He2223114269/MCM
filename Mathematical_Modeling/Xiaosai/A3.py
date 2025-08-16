import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
# 读取数据
dataset = np.array([[0.483, 1.50E+07, 1, 0],
                    [0.479, 1.70E+07, 0.991718427, 1],
                    [0.452, 2.00E+07, 0.935817805, 2],
                    [0.418, 2.50E+07, 0.865424431, 3],
                    [0.371, 3.00E+07, 0.768115942, 4],
                    [0.342, 3.30E+07, 0.708074534, 5],
                    [0.319, 3.50E+07, 0.660455487, 6],
                    [0.311, 3.70E+07, 0.64389234, 7],
                    [0.309, 3.70E+07, 0.639751553, 8]])

# 计算每天的生物降解速率常数
decayConstants = np.zeros((dataset.shape[0], 1))
for idx in range(1, dataset.shape[0]):
    decayConstants[idx] = np.log(dataset[idx-1, 0]/dataset[idx, 0])/(dataset[idx, 3]-dataset[idx-1, 3])

# 拟合生物降解速率常数，计算半衰期
fitCoefficients = np.polyfit(dataset[:, 3], np.log(dataset[:, 0]), 1)
fittedDecayConstant = -fitCoefficients[0]
halfLife = np.log(2)/fittedDecayConstant

# 计算有机物浓度随时间的变化
initialConcentration = dataset[0, 0]
initialMicrobeConcentration = dataset[0, 1]
time = np.linspace(0, 8, 100)
organicRatio = dataset[0, 2]
alpha = fittedDecayConstant * organicRatio * initialConcentration
organicConcentration = initialConcentration * np.exp(-alpha * time)

# 计算微生物浓度随时间的变化
microbeConcentration = initialMicrobeConcentration * np.exp(alpha * time)

# 计算每天的半衰期
halfLifePerDay = np.zeros(9)
halfLifePerDay[0] = 0  # 第0天的有机物降解的半衰期为0
for idx in range(1, 9):
    halfLifePerDay[idx] = np.log(2)/decayConstants[idx]

# 计算每天生物浓度变化快慢的变化
rateOfChange = np.zeros(9)
rateOfChange[0] = 0  # 第0天生物浓度变化快慢程度为0
for idx in range(1, 9):
    rateOfChange[idx] = dataset[idx, 1] - dataset[idx-1, 1]

# 绘制数据和拟合曲线-验证数据准确性-实际与理论差距情况
"""fig, axs = plt.subplots(3, 1, figsize=(8, 12))
axs[0].plot(dataset[:, 3], dataset[:, 0], 'k', markersize=10, linewidth=2)
axs[0].plot(time, organicConcentration, linewidth=2)
axs[0].set_xlabel('时间（天）', fontsize=14)
axs[0].set_ylabel('有机物浓度（mg/L）', fontsize=14)
axs[0].set_title(f'生物降解模型（a={fittedDecayConstant*organicRatio*initialConcentration:.4f}，t_{{1/2}}={halfLife:.2f}天）', fontsize=14)
axs[0].legend(['实验数据', '拟合曲线'], fontsize=14)

axs[1].plot(dataset[:, 3], dataset[:, 1], 'k-', markersize=10, linewidth=2)
axs[1].set_xlabel('时间（天）', fontsize=14)
axs[1].set_ylabel('微生物浓度（mg/L）', fontsize=14)
axs[1].set_title('微生物浓度变化', fontsize=14)

axs[2].plot(dataset[:, 3], dataset[:, 2], 'k-', markersize=10, linewidth=2)
axs[2].set_xlabel('时间（天）', fontsize=14)
axs[2].set_ylabel('有机物浓度比', fontsize=14)
axs[2].set_title('有机物浓度比变化', fontsize=14)
# axs[2].grid(True)"""
# 绘制有机物浓度随时间的变化
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(dataset[:, 3], dataset[:, 0], 'k', markersize=10, linewidth=2)
ax.plot(time, organicConcentration, linewidth=2)
ax.set_xlabel('时间（天）', fontsize=14)
ax.set_ylabel('有机物浓度（mg/L）', fontsize=14)
ax.set_title(f'生物降解模型（a={fittedDecayConstant*organicRatio*initialConcentration:.4f}，t_{{1/2}}={halfLife:.2f}天）', fontsize=14)
ax.legend(['实验数据', '拟合曲线'], fontsize=14)
plt.show()

# 绘制微生物浓度随时间的变化
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(dataset[:, 3], dataset[:, 1], 'k-', markersize=10, linewidth=2)
ax.set_xlabel('时间（天）', fontsize=14)
ax.set_ylabel('微生物浓度（mg/L）', fontsize=14)
ax.set_title('微生物浓度变化', fontsize=14)
plt.show()

# 绘制有机物浓度比随时间的变化
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(dataset[:, 3], dataset[:, 2], 'k-', markersize=10, linewidth=2)
ax.set_xlabel('时间（天）', fontsize=14)
ax.set_ylabel('有机物浓度比', fontsize=14)
ax.set_title('有机物浓度比变化', fontsize=14)
plt.show()
# 绘制半衰期与时间的关系曲线
fig, axs = plt.subplots(2, 1, figsize=(8, 8))
axs[0].plot(dataset[:, 3], halfLifePerDay, 'k-', markersize=10, linewidth=3)
axs[0].set_xlabel('时间（天）', fontsize=14)
axs[0].set_ylabel('半衰期（天）', fontsize=14)
axs[0].set_title('生物降解速率及有机物浓度比值变化分析', fontsize=14)

# 绘制生物降解速率常数与时间的关系曲线
axs[1].plot(dataset[:, 3], decayConstants, 'k-', markersize=10, linewidth=3)
axs[1].set_xlabel('时间（天）', fontsize=14)
axs[1].set_ylabel('生物降解常数', fontsize=14)

# 绘制生物浓度变化快慢与时间的关系曲线
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(dataset[:, 3], rateOfChange, 'k-', markersize=10, linewidth=3)
ax.set_xlabel('时间（天）', fontsize=14)
ax.set_ylabel('生物浓度变化快慢程度', fontsize=14)
ax.set_title('微生物浓度变化分析', fontsize=14)
plt.show()