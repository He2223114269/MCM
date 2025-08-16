import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA

# 读取数据
data = pd.read_csv('chenzhou.csv')

# 设置时间序列索引
data.index = pd.to_datetime(data['year'], format='%Y')
del data['year']

# 观察时间序列图
data.plot()
plt.show()

# 检验时序稳定性
from statsmodels.tsa.stattools import adfuller
result = adfuller(data)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
# p值大于0.05,拒绝原假设,说明时间序列是稳定的

# 划分训练数据与测试数据
train = data['2007':'2016']
test = data['2017':]

# ARIMA模型参数确定
# 定参数p, q的方法:断定ACF和PACF图,p与PACF chart的截尾点相关,q与ACF chart的截尾点相关
from statsmodels.tsa.stattools import acf, pacf
acf_chart = acf(train, nlags=20)
pacf_chart = pacf(train, nlags=20, alpha=0.05)

model = ARIMA(train, order=(2,1,2))
results = model.fit()

# 预测
predictions = results.predict(start=test.index[0],
                             end=test.index[-1],
                             dynamic=False)
# 观察预测效果
test.plot(legend=True)
predictions.plot(legend=True)
plt.title('Predictions vs actual')
plt.show()