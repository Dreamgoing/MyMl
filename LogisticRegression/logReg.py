"""
logistic regression working module
"""
from numpy import *
import matplotlib.pyplot as plt
import os


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open(r'./data/testSet.txt', 'r')

    """
    readlines(): Read and return the list of all logical lines remaining in the current file
    strip(): Return a copy of the string with the leading and trailing characters removed.
             If parameter is omitted or None,this function is to removing whitespace.
    split(): Return a list of the words in the string, using sep as the delimiter string.
    """
    for line in fr.readlines():
        # print(line.strip().split())
        lineArr = line.strip().split()
        # add x0 to dataMat
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def plotData(dataMat, labelMat, theta):
    dataArr = array(dataMat)
    n = shape(dataArr)[0]  # number of points to create
    # print(n)
    # print(dataArr)
    x0, y0, x1, y1 = [], [], [], []
    for i in range(n):
        if int(labelMat[i]) == 1:
            x1.append(dataArr[i, 1]), y1.append(dataArr[i, 2])
        else:
            x0.append(dataArr[i, 1]), y0.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(211)
    bx = fig.add_subplot(212)
    bx.plot(x0, y0, 'yo')
    # ax.scatter(x0,y0,s=30,c='red',marker='s')
    # ax.scatter(x1,y1,s=30,c='green')
    ax.plot(x0, y0, 'r.')
    ax.plot(x1, y1, 'bx')
    x = arange(-3.0, 3.0, 0.1)
    y = (-theta[0] - theta[1] * x) / theta[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def plotBestFit(theta):
    fig = plt.figure()
    x = arange(-3.0, 3.0, 0.1)
    y = (-theta[0] - theta[1] * x) / theta[2]
    plt.plot(x, y)
    plt.show()


def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


def gradAscent(dataMatIn, classLabels):
    """
    With gradient descent we're trying to minimize cost function rather than maximize it.
    :param dataMatIn:
    :param classLabels:
    :return: theta
    """
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)

    # learning rate
    alpha = 0.001

    # max loop number
    maxCycles = 500

    # y: theta0*x0+theta1*x1+theta2*x2
    theta = ones((n, 1))

    for k in range(maxCycles):
        h = sigmoid(dataMatrix * theta)
        # @todo to figure out why and when use gradient ascent
        error = (labelMat - h)
        # 经过了对thetaj求偏导数得出的公式
        theta += alpha * dataMatrix.transpose() * error
    return theta


if __name__ == '__main__':
    # print(os.curdir)
    data = loadDataSet()

    theta = gradAscent(data[0], data[1])

    listTheta = array(theta.transpose()).tolist()[0]
    plotData(data[0], data[1], listTheta)
