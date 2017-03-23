"""
linear regression working module
"""
import matplotlib.pyplot as plt
from numpy import *


def loadDataSet():
    # print('ok')
    dataMat = []
    labelMat = []
    fr = open(r'./data/ex1data1.txt')
    for line in fr.readlines():
        lineArr = line.strip().split(',')
        dataMat.append([1.0, float(lineArr[0])])
        labelMat.append(float(lineArr[1]))
    return dataMat, labelMat


def plotData(data, val, theta):
    n = shape(data)
    print(n)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(data, val, 'r.')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    x = arange(0, 24, 0.1)
    y = theta[0] + x * theta[1]
    ax.plot(x, y)
    plt.show()


def costFunction(data, val, theta):
    """

    :param data: matrix
    :param val: matrix
    :param theta: matrix
    :return: J
    """
    htheta = data * theta
    J = 0
    m = shape(data)[0]
    J = 1 / (2 * m) * ((htheta - val).transpose() * (htheta - val))
    return J


def featureNormalize(X):
    X_norm = array(X)

    #
    mu = zeros((1, shape(X)[1]))
    sigma = zeros((1,shape(X)[1]))

    # axis=0 对每一列进行运算
    mu = mean(X_norm,axis=0)
    sigma = std(X_norm,axis=0)

    #遍历每一列
    for i in range(X.shape[1]):
        X_norm[:,i] = (X_norm[:,i]-mu[i])/sigma[i] #归一化
    return X_norm,mu,sigma



def gradDescent(data, val):
    """

    :param dataMat: x 包含x0
    :param valMat: y
    :return: theta: parameter
    """
    dataMatrix = mat(data)
    valMatrix = mat(val).transpose()
    m, n = shape(dataMatrix)

    # learning rate
    alpha = 0.01

    # max loop number
    maxCycles = 1000

    # y: theta0*x0 + theta1*x1
    # # 包含theta0，且theta为列向量
    # theta = ones((1, n)) 这样定义为行向量
    theta = ones((n, 1))
    J_history = zeros((maxCycles, 1))
    for k in range(maxCycles):
        htheta = dataMatrix * theta
        # temp为暂存变量,下式为梯度下降法中，偏导数的公式
        temp = theta - alpha / m * dataMatrix.transpose() * (htheta - valMatrix)
        theta = temp
        J_history[k] = costFunction(dataMatrix, valMatrix, theta)
    return theta, J_history


def plotCostfunction(J_history):
    num = len(J_history)
    x = arange(1, num + 1)
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.plot(x, J_history)
    plt.xlabel(u'iteration times')
    plt.ylabel(u'cost value')
    plt.title(u'cost function')
    plt.show()


if __name__ == '__main__':
    """

    : data: 包含n个训练样本的列向量
    : val: 值
    """
    data, val = loadDataSet()
    # 提取list中的某一列
    dataX = [t[1] for t in data]
    # data = featureNormalize(data)
    tmp = gradDescent(data, val)
    theta = tmp[0].tolist()
    J_history = tmp[1].tolist()
    J_history = [t[0] for t in J_history]
    # print(J_history)
    # print(theta)
    theta = [t[0] for t in theta]
    # print(theta)
    # print(dataX)
    # plotData(dataX, val, theta)
    plotCostfunction(J_history)
