import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

df = pd.read_excel("data1.xlsx",header = None)

fig = plt.figure()
ax = plt.axes(projection="3d")

x = np.arange(0,4.02,0.02)
y = np.arange(0,5.02,0.02)
X, Y = np.meshgrid(x, y)
Z = np.array(df.values)
Z = -Z
Z1 = np.zeros((251,201))
ax.plot_surface(X,Y,Z,alpha=0.9, cstride=1, rstride = 1, cmap='rainbow')
ax.plot_surface(X,Y,Z1,alpha=0.9, cstride=1, rstride = 1, cmap='Blues')

plt.show()
