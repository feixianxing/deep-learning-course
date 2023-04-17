from course1.week3.utils.planar_utils import sigmoid
from course1.week3.utils.testCases import *


def layer_sizes(inner_X, inner_Y):
    """
    :param inner_X: 输入数据集，维度为(输入的数量，训练/测试的数量)
    :param inner_Y: 标签，维度为(输出的数量，训练/测试的数量)
    :return:
        n_x: 输入层单元的数量
        n_h: 隐藏层单元的数量
        n_y: 输出层单元的数量
    """

    n_x = inner_X.shape[0]
    n_h = 4
    n_y = inner_Y.shape[0]

    return n_x, n_h, n_y


# 测试layer_sizes
# print("====================测试layer_sizes====================")
# X_asses, Y_asses = layer_sizes_test_case()
# (n_x, n_h, n_y) = layer_sizes(X_asses, Y_asses)
# print("输入层的节点数量为: n_x = " + str(n_x))
# print("隐藏层的节点数量为: n_h = " + str(n_h))
# print("输出层的节点数量为: n_y = " + str(n_y))


# 初始化模型的参数
def initialize_parameters(inner_n_x, inner_n_h, inner_n_y):
    """
    :param inner_n_x: 输入层节点的数量
    :param inner_n_h: 隐藏层节点的数量
    :param inner_n_y: 输出层节点的数量
    :return:
        :parameters - 包含参数的字典:
            W1: 权重矩阵，维度为(n_h, n_x)
            b1: 偏置量，维度为(n_h, 1)
            W2: 权重矩阵，维度为(n_y, n_h)
            b2: 偏置量，维度为(n_y, 1)
    """
    np.random.seed(2)
    W1 = np.random.randn(inner_n_h, inner_n_x) * 0.01
    b1 = np.zeros(shape=(inner_n_h, 1))
    W2 = np.random.randn(inner_n_y, inner_n_h) * 0.01
    b2 = np.zeros(shape=(inner_n_y, 1))

    # 使用断言确保数据格式正确
    assert (W1.shape == (inner_n_h, inner_n_x))
    assert (b1.shape == (inner_n_h, 1))
    assert (W2.shape == (inner_n_y, inner_n_h))
    assert (b2.shape == (inner_n_y, 1))

    inner_parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    return inner_parameters


# 测试initialize_parameters
# print("====================测试initialize_parameters====================")
# n_x, n_h, n_y = initialize_parameters_test_case()
# parameters = initialize_parameters(n_x, n_h, n_y)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))


# 前向传播
def forward_propagation(inner_X, inner_param):
    """
    :param inner_X: 维度为(n_x, m)的输入数据
    :param inner_param: 初始化函数(initialize_parameters)的输出
    :return:
        A2: 使用sigmoid()函数计算的第二次激活后的数值
        cache: 包含"Z1", "A1", "Z2"和"A2"的字典类型变量
    """
    W1 = inner_param["W1"]
    b1 = inner_param["b1"]
    W2 = inner_param["W2"]
    b2 = inner_param["b2"]
    # 前向传播计算A2
    Z1 = np.dot(W1, inner_X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    inner_A2 = sigmoid(Z2)
    # 使用断言确保数据格式正确
    assert (inner_A2.shape == (1, inner_X.shape[1]))
    inner_cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": inner_A2
    }
    return inner_A2, inner_cache


# 测试forward_propagation
# print("====================测试forward_propagation====================")
# X_assess, parameters = forward_propagation_test_case()
# A2, cache = forward_propagation(X_assess, parameters)
# print(
#     np.mean(cache["Z1"]),
#     np.mean(cache["A1"]),
#     np.mean(cache["Z2"]),
#     np.mean(cache["A2"])
# )


# 计算损失(交叉熵损失)
def compute_cost(inner_A2, inner_Y, inner_param):
    """
    计算交叉熵成本
    :param inner_A2: 使用sigmoid()函数计算的第二次激活后的数值
    :param inner_Y: "True"标签向量，维度为(1, 数量)
    :param inner_param: 一个包含W1, B1, W2和B2的字典类型的变量
    :return:
        inner_cost: 交叉熵成本
    """
    m = inner_Y.shape[1]
    # W1 = inner_param["W1"]
    # W2 = inner_param["W2"]

    # 计算成本
    log_probs = np.multiply(np.log(inner_A2 + 1e-5), inner_Y) + np.multiply((1 - inner_Y), np.log(1 - inner_A2 + 1e-5))
    cost = -np.sum(log_probs) / m
    cost = float(np.squeeze(cost))

    assert (isinstance(cost, float))

    return cost


# 测试computed_cost
# print("====================测试computed_cost====================")
# A2, Y_assess, parameters = compute_cost_test_case()
# print("cost = " + str(compute_cost(A2, Y_assess, parameters)))


# 反向传播
def backward_propagation(inner_param, inner_cache, inner_X, inner_Y):
    """
    反向传播函数
    :param inner_param: 包含参数的一个字典类型变量
    :param inner_cache: 包含"Z1","A1","Z2","A2"的字典类型变量
    :param inner_X: 输入数据，维度为(2, 数量)
    :param inner_Y: “True”标签，维度为(1, 数量)
    :return:
        grads: 包含W和b的导数的一个字典类型变量
    """
    m = inner_X.shape[1]

    # W1 = inner_param["W1"]
    W2 = inner_param["W2"]

    A1 = inner_cache["A1"]
    inner_A2 = inner_cache["A2"]

    dZ2 = inner_A2 - inner_Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, inner_X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    inner_grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }

    return inner_grads


# 测试backward_propagation
# print("====================测试backward_propagation====================")
# parameters, cache, X_assess, Y_assess = backward_propagation_test_case()
# grads = backward_propagation(parameters, cache, X_assess, Y_assess)
# print("dW1 = " + str(grads["dW1"]))
# print("db1 = " + str(grads["db1"]))
# print("dW2 = " + str(grads["dW2"]))
# print("db2 = " + str(grads["db2"]))


# 更新参数
def update_parameters(inner_params, inner_grads, inner_learning_rate=1.2):
    """
    使用梯度下降更新规则更新参数
    :param inner_params: 包含参数的字典类型变量
    :param inner_grads: 包含导数的字典类型变量
    :param inner_learning_rate: 学习速率
    :return:
        parameters: 包含更新参数的字典类型变量
    """
    W1, W2 = inner_params["W1"], inner_params["W2"]
    b1, b2 = inner_params["b1"], inner_params["b2"]

    dW1, dW2 = inner_grads["dW1"], inner_grads["dW2"]
    db1, db2 = inner_grads["db1"], inner_grads["db2"]

    W1 = W1 - inner_learning_rate * dW1
    b1 = b1 - inner_learning_rate * db1
    W2 = W2 - inner_learning_rate * dW2
    b2 = b2 - inner_learning_rate * db2

    inner_updated_parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    return inner_updated_parameters


# 测试update_parameters
# print("====================测试update_parameters====================")
# parameters, grads = update_parameters_test_case()
# parameters = update_parameters(parameters, grads)
#
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))
