import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def hypothesis(theta: np.ndarray, x: np.ndarray):
    return np.dot(theta, x.T)


def compute_cost(theta: np.ndarray, x: np.ndarray, y: np.ndarray):
    h = np.dot(theta, x.T)
    inner = np.power((h.T - y), 2)
    return np.sum(inner) / (2 * len(x))


def gradient_descent(theta: np.ndarray, x: np.ndarray, y: np.ndarray, alpha, iters):
    temp = np.zeros(theta.shape)
    parameters = int(theta.shape[1])
    cost = np.zeros(iters)
    for i in range(iters):
        h = hypothesis(theta, x)
        error = h.T - y
        for j in range(parameters):
            term = np.multiply(error.T, x[:, j])
            s = np.sum(term)
            temp[0, j] = theta[0, j] - ((alpha / len(x)) * s)

        theta = temp
        cost[i] = compute_cost(theta, x, y)

    return theta, cost


def data1():
    # 导入数据（第一列：城市人口数量，第二列：该城市小吃店利润）
    data = pd.read_csv('./ex1data1.txt', header=None, names=['x', 'y'])
    data_h = data.head()
    print(data_h)
    # 展示数据
    data.plot(kind='scatter', x='x', y='y', figsize=(12, 8))
    plt.show()
    data.insert(0, 'x0', 1)
    data_h = data.head()
    print(data_h)
    print("================================================")

    cols = data.shape[1]
    x_set = data.iloc[:, :-1]  # X是data里的除最后列
    y_set = data.iloc[:, cols - 1:cols]  # y是data最后一列
    x = np.asarray(x_set.values)
    y = np.asarray(y_set.values)
    theta = np.asarray([[0, 0]])
    print('t shape: %s, x shape: %s, y shape: %s' % (theta.shape, x.shape, y.shape))
    print("================================================")
    g, cost = gradient_descent(theta, x, y, 0.001, 50)
    print('cost: %s' % cost)
    print("------------------------------------------------")
    print('g: %s' % g)
    print("================================================")
    nx = [0] * len(cost)
    ny = [0] * len(cost)
    for i in range(1, len(cost)+1):
        nx[i-1] = i
        ny[i-1] = cost[i-1]
    plt.xlabel("epoch")
    plt.ylabel("J(theta)")
    plt.plot(nx, ny)
    plt.show()
    test_x_l = np.linspace(data.x.min(), data.x.max(), 100).tolist()
    test_x = np.asarray([([1] * 100), test_x_l]).transpose()
    test_y = hypothesis(g, test_x)
    test_y_l = test_y[0].tolist()
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(test_x_l, test_y_l, 'r', label='Prediction')
    ax.scatter(data.x, data.y, label='Traning Data')
    ax.legend(loc=2)
    ax.set_xlabel('Population')
    ax.set_ylabel('Profit')
    ax.set_title('Predicted Profit vs. Population Size')
    plt.show()


def data2():
    data = pd.read_csv('./ex1data2.txt', header=None, names=['x1', 'x2', 'y'])
    data_h = data.head()

    data = (data - data.mean()) / data.std()
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

    theta = np.asarray([[0, 0, 0]])

    g, cost = gradient_descent(theta, x, y, 0.01, 500)
    print('cost: %s' % cost)
    print("------------------------------------------------")
    print('g: %s' % g)
    print("================================================")
    nx = [0] * len(cost)
    ny = [0] * len(cost)
    for i in range(1, len(cost) + 1):
        nx[i - 1] = i
        ny[i - 1] = cost[i - 1]
    plt.xlabel("epoch")
    plt.ylabel("J(theta)")
    plt.plot(nx, ny)
    plt.show()
    pass


def main():
    # data1()
    data2()
    pass


if __name__ == '__main__':
    main()
    # test()
