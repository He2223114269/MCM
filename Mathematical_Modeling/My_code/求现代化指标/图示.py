import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False
# 读取数据
df = pd.read_excel('data.xlsx')
print(df)
# 设置绘图风格
sns.set_style('darkgrid')
# 绘制折线图
x=['城乡协调', '环境治理', '经济建设', '民生保障', '文化科创']
sns.lineplot('城乡协调', y='年份', data=df)
# 添加标题
plt.title('现代化水平评价指标')
# 添加x轴和y轴标签
plt.show()


index = pd.date_range("1 1 2000", periods=100,
                      freq="m", name="date")
data = np.random.randn(100, 4).cumsum(axis=0)
wide_df = pd.DataFrame(data, index, ["a", "b", "c", "d"])
ax = sns.lineplot(data=wide_df)