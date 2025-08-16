import statsmodels.formula.api as smf
import pandas as pd
import random

# 读取数据
data = pd.read_excel('data2.xlsx')
print(data.head())

target = 'y'
variables = set(data.columns)
variables.remove(target)
variables.remove('年份/年')

# 定义数组存储添加/删除的变量
added_variables = []
deleted_variables = variables.copy()

# 随机生成初始模型
initial_model = random.sample(variables, 3)
for variable in initial_model:
    added_variables.append(variable)
    deleted_variables.remove(variable)

# 设置初始AIC值
aic = smf.ols(formula='{}~{}'.format(target, '+'.join(added_variables)), data=data).fit().aic
print('随机初始模型:{}~{}, AIC值为:{}'.format(target, '+'.join(added_variables), aic))
print()


# 添加变量
def add_variable():
    scores = []
    global best_add_score
    global best_add_variable
    print('添加变量:')
    for variable in deleted_variables:
        formula = '{}~{}'.format(target, '+'.join(added_variables + [variable]))
        score = smf.ols(formula=formula, data=data).fit().aic
        print('自变量{}: AIC值为{}'.format('+'.join(added_variables + [variable]), score))
        scores.append((score, variable))

    scores.sort(reverse=True)
    best_add_score, best_add_variable = scores.pop()
    print('最小AIC值为:', best_add_score)
    print()


# 删除变量
def delete_variable():
    scores = []
    global best_del_score
    global best_del_variable
    print('剔除变量:')
    for variable in added_variables:
        selected = added_variables.copy()
        selected.remove(variable)
        formula = '{}~{}'.format(target, '+'.join(selected))
        score = smf.ols(formula=formula, data=data).fit().aic
        print('自变量{}: AIC值为{}'.format('+'.join(selected), score))
        scores.append((score, variable))

    scores.sort(reverse=True)
    best_del_score, best_del_variable = scores.pop()
    print('最小AIC值为:', best_del_score)
    print()


print('剩余变量:', deleted_variables)
add_variable()
delete_variable()

while variables:

    # if aic < best_add_score < best_del_score or aic < best_del_score < best_add_score:
    #     print('当前回归方程为最优方程,{}~{},AIC值为:{}'.format(target, '+'.join(added_variables), aic))
    #     break
    if aic < best_add_score < best_del_score or aic < best_del_score < best_add_score:
        print('当前回归方程为最优方程,{}~{},AIC值为:{}'.format(target, '+'.join(added_variables), aic))

        # 输出回归方程的参数
        results = smf.ols(formula='{}~{}'.format(target, '+'.join(added_variables)), data=data).fit()
        print(results.params)
        break
    elif best_add_score < best_del_score < aic or best_add_score < aic < best_del_score:
        print('目前最小AIC值为:', best_add_score)
        print('选择自变量:', '+'.join(added_variables + [best_add_variable]))
        print()
        deleted_variables.remove(best_add_variable)
        added_variables.append(best_add_variable)
        print('剩余变量:', deleted_variables)
        aic = best_add_score
        add_variable()
    else:
        print('当前最小AIC值为:', best_del_score)
        print('需要剔除的变量:', best_del_variable)
        aic = best_del_score
        added_variables.remove(best_del_variable)
        delete_variable()

