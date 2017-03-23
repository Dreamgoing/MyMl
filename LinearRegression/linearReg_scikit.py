import numpy as np
# 导入线性模型
from sklearn import linear_model, datasets

# 导入归一化包
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def testLinearModel():
    # Load the diabetes dataset
    diabetes = datasets.load_diabetes()


    # Use only one feature
    diabetes_X = diabetes.data[:, np.newaxis, 2]

    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]

    # Split the targets into training/testing sets
    diabetes_y_train = diabetes.target[:-20]
    diabetes_y_test = diabetes.target[-20:]

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(diabetes_X_train, diabetes_y_train)

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f"
          % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

    # Plot outputs
    plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
    plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue',
             linewidth=3)

    """
    xticks() 设置x坐标的标签
    yticks() 设置y坐标的标签
    """
    plt.xticks(())
    plt.yticks(())

    plt.show()


def linearReg():
    print('loading dataSet...')
    data = np.loadtxt(r'./data/ex1data1.txt', dtype=np.float64, delimiter=',')

    # X 对应0到倒数第2列
    # print(data[:,0])
    X = np.array(data[:, 0:-1], dtype=np.float64)

    # Y 对应倒数第1列
    Y = np.array(data[:, -1], dtype=np.float64)

    # 进行归一化操作
    scaler = StandardScaler()
    scaler.fit(X)

    # 归一化，训练集中的x数据
    x_train = scaler.transform(X)


    # print(x_train)
    # x_test = scaler.transform(np.array([1650.0, 3.0]))
    # print(x_test)

    # 创建一个线性回归的模型
    linReg = linear_model.LinearRegression()

    linReg.fit(x_train,Y)
    # print(linReg.coef_)
    plt.scatter(x_train,Y,s=3)
    plt.plot(x_train,linReg.predict(x_train))
    # print(data)
    print(x_train)
    print(Y)
    plt.show()


class Employee:
    # empCount 为一个类变量，可以在这个类的所有实例之间共享
    empCount = 0

    def __init__(self, name, salary):
        self.name = name
        self.salary = salary
        self.empCount += 1

def testEmployee():
    a = Employee('abc','111')
    print(a.name)


if __name__ == '__main__':
    # testLinearModel()
    linearReg()
    # testEmployee()
