# 导入线多项式拟合工具：多项式拟合
import numpy as np
import matplotlib.pyplot as plt
from numpy import polyfit, poly1d,polyval

# X = np.arange(2011, 2021)
# Y = np.array([1.3, 1.36, 1.41, 1.51, 1.64, 1.64, 1.5, 1.56, 1.6, 1.63, ])
# new_x = np.array([2007, 2008, 2009, 2010, ])

# 数据准备
X = np.arange(2011, 2021)
# Y = np.array([1.3, 1.36, 1.41, 1.51, 1.64, 1.64, 1.5, 1.56, 1.6, 1.63, ])# 每一农村劳动力负担系数/人
Y = np.array([47.03,48.5,50.34,52.22,53.8,54.88,56.04,58.18,59.04,59.53,])#城镇化率/%


# Y = np.array([1.53,1.89,1.81,1.53,1.45,1.45,1.39,1.51,1.54,1.51])# 每一城镇就业者负担系数/人
new_x = np.array([2007, 2008, 2009, 2010, 2011,2012])
coeff = polyfit(X, Y,4)

new_y = polyval(coeff, new_x)
print
p = plt.plot(X, Y, 'rx')
p = plt.plot(X, coeff[0] * X + coeff[1], 'k-')
p = plt.plot(X, Y, 'b--')
p = plt.plot(new_x, new_y, 'r-o')
print(f"new_y: {new_y}")
plt.show()

# 还可以用 poly1d 生成一个以传入的 coeff 为参数的多项式函数：
# f = poly1d(coeff)
# p = plt.plot(x, noise_y, 'rx')
# p = plt.plot(x, f(x))