import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
from statsmodels.stats.outliers_influence import variance_inflation_factor

data = pd.read_excel('data2.xlsx')
y = data['y']
X = data[['x2','x3']]



# print(X)
# print(y)
# 标准化自变量X
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# print(X_std)

model = sm.OLS(y, X_std).fit()
"""
model.params:回归系数,表示各自变量对因变量y的影响程度。
model.tvalues:t值,用于判断各自变量在模型中的显著性。绝对值大于2时自变量显著。
model.rsquared:决定系数R^2,表示模型的拟合优度。值越大,模型解释因变量变化的能力越强。
model.rsquared_adj:调整后决定系数,综合考虑自变量数量对rsquared的影响。
model.fvalue:F值,用于同时检验自变量在模型中的总体显著性。值越大,总体显著性越高。
model.pvalues:P值,同样用于判断各自变量及总体显著性。小于0.05时显著。
model.conf_int():置信区间,表示在一定置信水平下,回归系数的估计范围。包含0则不显著。
model.bse:回归系数的标准误,用于判断回归系数估计的精度。值越小,估计精度越高。
"""
print(f"回归系数params:--------------\n{model.params} ")
print(f"t值tvalues:--------------\n--------------\n {model.tvalues}")
print(f"F值rsquared:--------------\n {model.rsquared}")
print(f"P值rsquared_ad:--------------\n {model.rsquared_adj}")
print(f"置信区间fvalue: {model.fvalue}")
print(f"pvalues:--------------\n {model.pvalues}")
print(f"conf_int:--------------\n {model.conf_int()}")
print(f"bse:--------------\n {model.bse}")

# 预测值
y_pred = model.predict(X_std)

# 残差
residuals = y - y_pred

# 输出模型参数与残差
print(model.params)
print(residuals)

# 残差图
plt.scatter(y_pred, residuals)
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()
# 残差自相关系数
print(np.corrcoef(residuals, y_pred))
# 残差均值
print(np.mean(residuals))
# 残差标准差
print(np.std(residuals, ddof=1))
# 残差方差
print(np.var(residuals, ddof=1))
