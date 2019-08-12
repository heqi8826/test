import numpy as np

# 导入数据集
points = np.genfromtxt('data.csv', delimiter=',')
print(len(points[:, 0]))
print(points.shape[0])


# 计算loss损失
def myloss(points, w, b):
    loss_sum = 0

    for i in range(len(points[:, 0])):
        x = points[i][0]
        y = points[i][1]
        loss_sum += (w * x + b - y) ** 2
    return loss_sum / points.shape[0]


# 计算 w b
def mygradient(points, w_current, b_current, lr):
    w_gradient = 0
    b_gradient = 0
    N = len(points[:, 0])
    for i in range(points.shape[0]):
        # print(len(points[:, 0])) 测试打印长度临时使用
        x = points[i, 0]
        y = points[i, 1]
        # grad_w = 2(wx+b-y)*x
        w_gradient += (2/N) * x * ((w_current * x + b_current) - y)
        # grad_b = 2(wx+b-y)
        b_gradient += (2/N) * ((w_current * x + b_current) - y)
    new_w = w_current - lr * w_gradient
    new_b = b_current - lr * b_gradient
    return [new_w, new_b]


# 最佳w 和 b 预测
def best_w_b(points, w_start, b_start, num, lr):
    w = w_start
    b = b_start
    # 迭代数次 得出最佳w b
    for i in range(int(num)):
        [w, b] = mygradient(points, w, b, lr)
        return [w, b]


# 运行函数runner
def myrunner():
    lr = 0.0001
    initial_b = 0  # initial y-intercept guess
    initial_w = 0  # initial slope guess
    num = 1000
    print("Starting gradient descent at b = {0}, w = {1}, error = {2}"
          .format(initial_b, initial_w,
                  myloss(points, initial_w, initial_b))
          )
    print("Running...")
    [w, b] = best_w_b(points, initial_w, initial_b, num, lr)
    print("After {0} iterations  w = {2}, b = {1}, error = {3}".
          format(num, w, b,
                 myloss(points, w, b))
          )


myrunner()





