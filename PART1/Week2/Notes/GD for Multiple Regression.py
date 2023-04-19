import numpy as np
import matplotlib.pyplot as plt

# 获得训练数据
x_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

m, n = x_train.shape


# 获得cost function
def cost_function(x, y, w, b):
    cost = 0

    for i in range(m):
        y_temp = np.dot(x[i], w) + b
        cost = cost + (y_temp - y[i]) ** 2

    return cost / (2 * m)


# 获得 w 与 b 的偏导
def compute_gradient(x, y, w, b):
    dw = np.zeros((n,))
    db = 0

    for i in range(m):
        delta = np.dot(x[i], w) + b - y[i]
        for j in range(n):
            dw[j] = dw[j] + delta * x[i][j]
        db = db + delta

    return dw / m, db / m


# 开始计算
def gradient_descent(x, y, w_init, b_init, times, alpha):
    w = w_init
    b = b_init
    J_history = [cost_function(x, y, w, b)]

    for i in range(times):
        dw, db = compute_gradient(x, y, w, b)
        w = w - alpha * dw
        b = b - alpha * db

        J_history.append(cost_function(x, y, w, b))

    return w, b, J_history


w_init = np.zeros((n,))
b_init = 0

times = 2500
alpha = 8e-10

w_finnal, b_finnal, J_history = gradient_descent(x_train, y_train, w_init, b_init, times, alpha)

print(J_history)
time = range(len(J_history))
plt.plot(time, J_history)
plt.show()



