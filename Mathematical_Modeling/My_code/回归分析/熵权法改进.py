import pandas as pd
import numpy as np
import math

def normalize(series):  # 正向指标归一化 减最小值的min-max方法
    minimum = series.min()
    maximum = series.max()
    normalized = []
    for value in series:
        normalized.append((value - minimum) / (maximum - minimum))
    return pd.Series(normalized, name=series.name)


def normalize_negative(series):  # 负向指标归一化
    minimum = series.min()
    maximum = series.max()
    normalized = []
    for value in series:
        normalized.append((maximum - value) / (maximum - minimum))
    return pd.Series(normalized, name=series.name)


def normalization(df, indicators):
    normalized_df = pd.DataFrame()
    for column in df.columns:
        if indicators[column] > 0:
            normalized_df = pd.concat([normalized_df, normalize(df[column])], axis=1)
        else:
            normalized_df = pd.concat([normalized_df, normalize_negative(df[column])], axis=1)
    return normalized_df

def calculate_pij(df):
    result = df.copy()
    for column in range(df.shape[1]):
        sum_of_column = df.iloc[:, column].sum()
        for row in range(df.shape[0]):
            result.iloc[row, column] = df.iloc[row, column] / sum_of_column
    return result

def calculate_entropy(series):
    length = len(series)
    def log(x):
        if x > 0:
            return math.log(x)
        else:
            return 0
    entropy = 0
    for value in series:
        entropy += value * log(value)
    return -(1 / log(length)) * entropy

def calculate_info(dfij):
    result = dfij.copy()
    weights = pd.DataFrame(index=result.columns, dtype='float64')
    entropies = []
    for column in result.columns:
        entropies.append(calculate_entropy(result[column]))
    weights['熵'] = entropies
    weights['差异性系数'] = 1 - np.array(entropies)
    total = weights['差异性系数'].sum()
    weight = []
    for value in weights['差异性系数']:
        weight.append(value / total)
    weights['权重'] = weight
    return weights

def calculate_output(dfn, weights):
    weights = weights['权重']
    return np.matmul(dfn, weights)

df = pd.read_excel('文化投入指标汇总.xlsx', header=None)
print("===============文化投入指标汇总=================")
indicators = [1,1,1,1]
normalized_df = normalization(df, indicators)
dfij = calculate_pij(normalized_df)
weights = calculate_info(dfij)
normalized_df = normalized_df.set_index(df.index, drop=True)
# print(normalized_df)
print(weights)
print("标准化权重")
output = calculate_output(normalized_df, weights)
print(output)
print("原始权重")
output = np.matmul(df, weights['权重'])
print(output)

df = pd.read_excel('环境投入指标汇总.xlsx', header=None)
print("===============环境投入指标汇总=================")
indicators = [1,1,1,1]
normalized_df = normalization(df, indicators)
dfij = calculate_pij(normalized_df)
weights = calculate_info(dfij)
normalized_df = normalized_df.set_index(df.index, drop=True)
# print(normalized_df)
print(weights)
print("标准化权重")
output = calculate_output(normalized_df, weights)
print(output)
print("原始权重")
output = np.matmul(df, weights['权重'])
print(output)

df = pd.read_excel('社会投入指标汇总.xlsx', header=None)
print("===============社会投入指标汇总=================")
indicators = [1,1,1,1,1,1,1,1,]
normalized_df = normalization(df, indicators)
dfij = calculate_pij(normalized_df)
weights = calculate_info(dfij)
normalized_df = normalized_df.set_index(df.index, drop=True)
# print(normalized_df)
print(weights)
print("标准化权重")
output = calculate_output(normalized_df, weights)
print(output)
print("原始权重")
output = np.matmul(df, weights['权重'])
print(output)

df = pd.read_excel('经济投入指标汇总.xlsx', header=None)
print("===============经济投入指标汇总=================")
indicators = [1,1,1,1,1,1,1,1,1,1,1,1]
normalized_df = normalization(df, indicators)
dfij = calculate_pij(normalized_df)
weights = calculate_info(dfij)
normalized_df = normalized_df.set_index(df.index, drop=True)
# print(normalized_df)
print(weights)
print("标准化权重")
output = calculate_output(normalized_df, weights)
print(output)
print("原始权重")
output = np.matmul(df, weights['权重'])
print(output)
