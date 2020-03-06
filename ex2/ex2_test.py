import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

def sort_m(x, y, fs, f):
    out = []
    for i in range(len(y)):
        for j in range(len(y[i])):
            if fs[i][j] == f:
                out.append([x[i][j], y[i][j]])
    return out


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def cost(theta, x, y):
    theta = np.asmatrix(theta)
    x = np.asmatrix(x)
    y = np.asmatrix(y)
    left = np.multiply(y, np.log(sigmoid(x * theta.T)))
    right = np.multiply(1 - y, np.log(1 - sigmoid(x * theta.T)))
    return np.sum(left + right) / -(len(x))


def gradient_descent(theta, x, y, alpha, epoch):
    theta = np.asmatrix(theta)
    x = np.asmatrix(x)
    y = np.asmatrix(y)
    m, n = np.shape(x)
    c = [0] * epoch
    for i in range(epoch):
        h = sigmoid(x * theta.T)
        error = x.T * (h - y) / m
        theta = theta - alpha * error.T
        c[i] = cost(theta, x, y)
    return theta, c


def feature_scale(data: np.ndarray) -> np.ndarray:
    """ 特征缩放
    Args:
        data: 特征数据

    Returns:
        缩放后的数据
    """
    m, n = data.shape
    mean = np.sum(data) / (m * n)
    data_max = np.max(data)
    data_min = np.min(data)
    difference = data_max - data_min
    return (data - mean) / data.std()

def feature_sort(x1: np.ndarray, x2: np.ndarray , y: np.ndarray, f) -> list :
    """ 特征数据分类筛选
    只支持二维数据
    Args:
        x1: 特征数据1
        x2: 特征数据2
        y: 结果数据1
        f: 类别

    Returns:
        指定分类的特征数据
    """
    m, n = y.shape
    out = []
    for i in range(m):
        for j in range(n):
            if y[i][j] == f:
                out.append([x1[i][j], x2[i][j]])
    return out
def data1():
    # 导入数据（第一列：城市人口数量，第二列：该城市小吃店利润）
    data = pd.read_csv('./ex2data1.txt', header=None, names=['x1', 'x2', 'y'])
    data.insert(0, 'x0', 1)
    data_h = data.head()
    print(data_h)
    data_list = data.values.tolist()
    x_1 = [[i[1] for i in data_list]]
    x_2 = [[i[2] for i in data_list]]
    y_list = [[i[3] for i in data_list]]
    c_0 = sort_m(x_1, x_2, y_list, 0)
    c_1 = sort_m(x_1, x_2, y_list, 1)
    # 拆分数据，并画图
    c_0_x = [[i[0] for i in c_0]]
    c_0_y = [[i[1] for i in c_0]]
    c_1_x = [[i[0] for i in c_1]]
    c_1_y = [[i[1] for i in c_1]]
    plt.figure(figsize=(10, 7))
    plt.subplot(221)
    plt.scatter(c_0_x, c_0_y, label='c0', c='b', marker='.')
    plt.scatter(c_1_x, c_1_y, label='c1', c='g', marker='.')
    plt.legend()
    x_f = data.iloc[:, :-1]
    cols = data.shape[1]
    y_f = data.iloc[:, cols - 1:cols]
    x = x_f.values
    y = y_f.values
    theta = np.asarray([[0] * (cols - 1)])
    c1 = cost(theta, x, y)
    print('========================')
    print('first theta: %s' % theta)
    print('first cost: %s' % c1)
    start_time = int(round(time.time() * 1000))
    # thetas, errors = gradient_descent(theta, x, y, 0.001, 1000000)
    thetas, errors = gradient_descent(theta, x, y, 0.001, 500000)
    end_time = int(round(time.time() * 1000))
    c2 = cost(thetas, x, y)
    print('========================')
    print('last theta: %s' % thetas)
    print('last cost: %s' % c2)

    print('spend: %ds' % (int((end_time - start_time)/1000)))
    nx = [0] * len(errors)
    ny = [0] * len(errors)
    for i in range(1, len(errors) + 1):
        nx[i - 1] = i
        ny[i - 1] = errors[i - 1]
    plt.subplot(222)
    plt.xlabel("epoch")
    plt.ylabel("J(theta)")
    plt.plot(nx, ny)
    # 决策边界
    plt.subplot(223)
    plt.scatter(c_0_x, c_0_y, label='c0', c='b', marker='.')
    plt.scatter(c_1_x, c_1_y, label='c1', c='g', marker='.')
    new_x = np.linspace(30, 100, 100)
    t_list = thetas.tolist()
    new_y = (-t_list[0][0] - t_list[0][1] * new_x) / t_list[0][2]
    plt.plot(new_x, new_y, c='y', label='Prediction')
    plt.legend()
    plt.show()
    pass


def data2():
    data = pd.read_csv('./ex1data2.txt', header=None, names=['x1', 'x2', 'y'])
    data_h = data.head()
    print(data_h)
    print("================================================")
    x1_set = data.iloc[:, 0:1]
    x2_set = data.iloc[:, 1:2]
    x1 = x1_set.values
    x2 = x2_set.values
    data.insert(0, 'x0', 1)
    data_h = data.head()
    print(data_h)
    print("================================================")
    cols = data.shape[1]
    x_set = data.iloc[:, :-1]
    y_set = data.iloc[:, cols - 1:cols]
    x = x_set.values
    y = y_set.values
    plt.scatter(x1, y, marker='.', c='b')
    plt.scatter(x2, y, marker='.', c='c')

    plt.show()
    pass


def plot_feature_group(features: np.ndarray, result_set: np.ndarray):
    features_m, features_n = features.shape
    x1 = features[:, :1]
    x2 = features[:, features_n - 1:features_n]
    c_0 = feature_sort(x1, x2, result_set, 0)
    c_1 = feature_sort(x1, x2, result_set, 1)
    c_0_x = [[i[0] for i in c_0]]
    c_0_y = [[i[1] for i in c_0]]
    c_1_x = [[i[0] for i in c_1]]
    c_1_y = [[i[1] for i in c_1]]
    plt.scatter(c_0_x, c_0_y, label='c0', c='b', marker='.')
    plt.scatter(c_1_x, c_1_y, label='c1', c='g', marker='.')
    plt.legend()


def plot_effect(errors):
    epoch = len(errors)
    nx = [0] * epoch
    ny = [0] * epoch
    for i in range(1, epoch + 1):
        nx[i - 1] = i
        ny[i - 1] = errors[i - 1]
    plt.xlabel("epoch")
    plt.ylabel("J(theta)")
    plt.plot(nx, ny)

def test():
    plt.figure(figsize=(10, 7))
    # 导入数据
    data = pd.read_csv('./ex2data1.txt', header=None, names=['x1', 'x2', 'y'])
    data_arr = data.values
    data_m, data_n = data_arr.shape
    features = data_arr[:, :-1]
    result_set = data_arr[:, data_n-1:data_n]
    plt.subplot(221)
    plot_feature_group(features, result_set)
    # 特征缩放
    features = feature_scale(features)
    plt.subplot(222)
    plot_feature_group(features, result_set)
    # 插入偏置值
    features = np.insert(features, 0, 1, 1)
    features_m, features_n = features.shape
    theta = np.asarray([[0] * features_n])
    start_time = int(round(time.time() * 1000))
    thetas, errors = gradient_descent(theta, features, result_set, 0.01, 10000)
    end_time = int(round(time.time() * 1000))
    c2 = cost(thetas, features, result_set)
    print('Base thetas: %s, error: %s, spend(s): %.4f' % (thetas, c2, (end_time - start_time)/1000))
    plt.subplot(223)
    plot_effect(errors)
    features = features[:, 1:3]
    plt.show()
    pass


def main():
    data1()
    # data2()
    pass


if __name__ == '__main__':
    # main()
    test()
