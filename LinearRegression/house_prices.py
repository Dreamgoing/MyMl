"""
House prices problem in Kaggle
"""
import inline as inline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
# %matplotlib inline
warnings.filterwarnings('ignore')

def loadData():
    """

    :return:DataFrame: the data structure in pandas.
    """
    X_train = pd.read_csv(r'./data/train.csv')

    return X_train

# 单变量分析
def univariableAnalysis(X):
    """

    :param X: dataSet
    :return: none
    """

    # Histograms直方图
    sns.distplot(X['SalePrice'])
    sns.plt.show()
    """
    由图数据分析得到三条结论
    1. 偏离正态分布
    2. 有明显的正向偏移
    3. 有峰态
    """

    # skewness and kurtosis
    """
    显示了偏斜度和峰度
    """

    print("Skewness: %f" % X['SalePrice'].skew())
    print("Kurtosis: %f" % X['SalePrice'].kurt())

# 多变量分析（多个变量之间的关系）
def multivarableAnalysis0(X):
    """
    :@detail: relationship between numerical variables.
    :param X: dataSet dtype: DataFrame
    :return: none
    """

    # 设置其他因素与价格之间的关系
    var = 'GrLivArea'
    """
    根据图表分析可得
    GrLivArea 与 SalePrice 之间有 线性关系(linear relationship)
    """

    # var = 'TotalBsmtSF'
    """
    concat函数说明
    pandas.concat(objs,axis,...)
    :param
        objs: a sequence or mapping of Series, DataFrame, or Panel objects
        axis: {0/’index’, 1/’columns’}, default 0
              The axis to concatenate along
    :return concatenated : type of objects


    """

    data = pd.concat([X['SalePrice'], X[var]],axis=1)
    print(data)
    # data.plot.scatter(x=var,y='SalePrice',ylim=(0,800000))


    # 显示图表
    sns.plt.show()

def multivarableAnalysis1(X):
    """
    :@detail: relationship between categorical variables.
    :param X: dataSet dtype: DataFrame
    :return: none
    """

    # box plot overallqual/saleprice
    var = 'OverallQual'
    data = pd.concat([X['SalePrice'],X[var]],axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x=var, y="SalePrice", data=data)
    fig.axis(ymin=0, ymax=800000)


    var = 'YearBuilt'
    data = pd.concat([X['SalePrice'], X[var]], axis=1)
    f, ax = plt.subplots(figsize=(16, 8))
    fig = sns.boxplot(x=var, y="SalePrice", data=data)
    fig.axis(ymin=0, ymax=800000)


    # xticks 函数用来修改x标签
    plt.xticks(rotation=90)

    sns.plt.show()

def correlationMatrix(X):
    """
    @todo: there are some bug in this function
    :param X:
    :return:
    """
    corrmat = X.corr()
    # print(corrmat)
    # f,ax = plt.subplot(figsize=(16,8))
    sns.heatmap(corrmat,vmax=.8,square=True)
    # plt.xticks(rotaton=90)


    # saleprice correlation matrix
    k = 10  # number of variables for heatmap
    cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
    cm = np.corrcoef(X[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,
                     xticklabels=cols.values)
    plt.show()
    plt.xticks(rotation=90)
    sns.plt.show()

if __name__ == '__main__':
    X = loadData()
    # columns类似于sql table里表的列名
    # print(X.columns)

    # print(X['SalePrice'].describe())
    # univariableAnalysis(X)
    # multivarableAnalysis0(X)
    # multivarableAnalysis1(X)
    correlationMatrix(X)