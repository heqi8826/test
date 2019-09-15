import numpy as np


points = np.genfromtxt('./data.csv', dtype=np.float32, delimiter=',')
# print(points, points[0, 0], points[0, 1], len(points))


# y = wx+b
def loss(w, b, points):
    loss = 0
    for i in range(len(points)):
        range(len(points))
        x = points[i, 0]
        y = points[i, 1]
        loss_current = (w * x + b - y)**2
        loss += loss_current
    return loss


def get_gradient_w_b(w_current, b_current, points, lr):
    # 初始化w_gradient 和 b_gradient
    w_gradient = 0
    b_gradient = 0
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]

        # 计算loss对w的导数 和 loss 对b的导数。
        w_gradient += 2/len(points) * (w_current * x + b_current - y) * x
        b_gradient += 2/len(points) * (w_current * x + b_current - y)
    # 计算最优导数w b ，什么叫最优，所有points点上的导数与初始的w b的变化率
    w = w_current - lr * w_gradient
    b = b_current - lr * b_gradient
    return w, b


def times_get_w_b(w, b, points, lr, times):
    w_current = 0
    b_current = 0
    for i in range(times):
        w, b = get_gradient_w_b(w_current, b_current, points, lr)
    return w, b


def main(points):
    lr = 0.0001
    w = 0
    b = 0
    times = 1000
    for i in range(len(points)):
        loss_result = loss(w, b, points)
        w_new, b_new = times_get_w_b(w, b, points, lr, times)
        print('计算的第{0}次的损失值loss等于{1},参数w={2},b={3}'.format(i, loss_result, w_new, b_new))


if __name__ == '__main__':
    main(points)
