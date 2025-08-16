import pandas as pd
import numpy as np
import math


def nml(series):  # 正向指标归一化 减最小值的min-max方法
    l = []
    for i in series:
        l.append((i - series.min()) / (series.max() - series.min()))
    return pd.Series(l, name=series.name)


def nml_max(series):  # 负向指标归一化
    l = []
    for i in series:
        l.append((series.max() - i) / (series.max() - series.min()))
    return pd.Series(l, name=series.name)


def nmlzt(df, nml_nmlmax):  # 归一化函数，对正负向指标分别调用nml()和nml_max()
    dfn = pd.DataFrame()
    for i in df.columns:
        if (nml_nmlmax[i] > 0):
            dfn = pd.concat([dfn, nml(df[i])], axis=1)
        else:
            dfn = pd.concat([dfn, nml_max(df[i])], axis=1)

    # dfn为归一化的数据
    return dfn


def pij(df):  # 求信息熵公式中的p，这里直接用取值除以取值总和，而不是数量的比例
    D = df.copy()
    for i in range(D.shape[1]):  # 列
        sum = D.iloc[:, i].sum()
        for j in range(D.shape[0]):  # 行
            D.iloc[j, i] = D.iloc[j, i] / sum
            # 算pij
    return D


def entropy(series):  # 计算信息熵
    _len = len(series)

    def ln(x):
        if x > 0:
            return math.log(x)
        else:
            return 0

    s = 0
    for i in series:
        s += i * ln(i)
    return -(1 / ln(_len)) * s


def _result(dfij):  # 求e、d、w并返回
    dfn = dfij.copy()
    w = pd.DataFrame(index=dfn.columns, dtype='float64')
    l = []
    for i in dfn.columns:
        l.append(entropy(dfn[i]))
    w['熵'] = l
    w['差异性系数'] = 1 - np.array(l)
    sum = w['差异性系数'].sum()
    l = []
    for i in w['差异性系数']:
        l.append(i / sum)
    w['权重'] = l
    return w


def out(dfn, w):
    w = w['权重']
    answer = np.matmul(dfn, w)
    return answer

"""
# 城乡协调
# sheet = ['城乡协调_1.xlsx', ]
# for i in sheet:

# df = pd.read_excel('城乡协调_1.xlsx',header=None)	#读取你需要计算的文件
# nml_nmlmax = [0,1,0,0]

# df = pd.read_excel('环境治理_1.xlsx',header=None)	#读取你需要计算的文件
# nml_nmlmax = [0,1,1,0]

# df = pd.read_excel('经济建设_1.xlsx',header=None)	#读取你需要计算的文件
# nml_nmlmax = [1,1,1,1,1,1,1,1,1]

# df = pd.read_excel('民生保障_1.xlsx', header=None)  # 读取你需要计算的文件
# nml_nmlmax = [1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1]


# df = pd.read_excel('文化科创_1.xlsx', header=None)  # 读取你需要计算的文件
# nml_nmlmax = [1, 1, 1, 1]

# df = pd.read_excel('现代化指标.xlsx', header=None)  # 读取你需要计算的文件
# nml_nmlmax = [1, 1, 1, 1, 1]
"""
df = pd.read_excel('所有数据_1.xlsx', header=None)  # 读取你需要计算的文件
nml_nmlmax = [0.00, 1, 0, 0, 0, 1, 1, 0, 1.00, 1, 1, 1, 1.00, 1, 1, 1, 1, 1.00, 0.00, 1, 1, 0, 1, 1.00, 1, 1, 1, 1, 1,
              1.00, 1.00, 1.00, 1.00,]

print(df)
dfn = nmlzt(df, nml_nmlmax)  # 归一化
dfij = pij(dfn)  # 求p
w = _result(dfij)  # 求权重
# w.to_excel('weight_info_entropy.xlsx', sheet_name='权重')#输出结果
dfn = dfn.set_index(df.index, drop=True)
print(dfn)
print(w)

answer = out(dfn, w)
print(answer)

answer = np.matmul(df, w['权重'])
print(answer)

