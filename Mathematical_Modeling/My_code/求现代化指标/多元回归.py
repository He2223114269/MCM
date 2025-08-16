import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import matplotlib
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False

data = pd.read_excel('data.xlsx')
y = data['y']
X = data[['x16','x20','x10','x18','x26','x23','x7','x14','x11','x12','x9','x24','x25','x22','x8']]

# 使用随机森林模型进行预测
model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=0)

# 交叉验证评估模型性能
scores = cross_val_score(model, X, y, cv=5)
print('交叉验证得分:', scores)
print('平均得分:', np.mean(scores))

# 拟合模型
model.fit(X,y)

# 预测值
y_pred = model.predict(X)

# 残差
residuals = y - y_pred

# 残差图
plt.figure(figsize=(8,6))
plt.scatter(y_pred, residuals)
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.savefig('图片.png')
plt.show()

# 残差自相关系数
print(np.corrcoef(residuals, y_pred))
# 残差均值
print(np.mean(residuals))
# 残差标准差
print(np.std(residuals, ddof=1))
# 残差方差
print(np.var(residuals, ddof=1))

# 输出模型参数与残差
print("feature_importances_")
print(model.feature_importances_)
print(model.score(X, y))
print(residuals)

# 输出特征的方差膨胀因子
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
print(vif)