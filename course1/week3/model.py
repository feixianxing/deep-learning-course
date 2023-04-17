from steps import *
from course1.week3.utils.testCases import *


# 整合
def nn_model(inner_X, inner_Y, inner_n_h, num_iteration, learning_rate, print_cost=False):
    """
    :param inner_X: 数据集，维度为(2, 示例数)
    :param inner_Y: 标签，维度为(1, 示例数)
    :param inner_n_h: 隐藏层的数量
    :param num_iteration: 梯度下降循环中的迭代次数
    :param learning_rate: 学习率
    :param print_cost: 如果为True，则每1000次迭代打印一次成本数值
    :return:
        parameters: 模型训练后的参数，可以用来进行预测
    """
    np.random.seed(3)
    inner_n_x = layer_sizes(inner_X, inner_Y)[0]
    inner_n_y = layer_sizes(inner_X, inner_Y)[2]

    inner_params = initialize_parameters(inner_n_x, inner_n_h, inner_n_y)
    # W1 = inner_params["W1"]
    # b1 = inner_params["b1"]
    # W2 = inner_params["W2"]
    # b2 = inner_params["b2"]

    for i in range(num_iteration):
        inner_A2, inner_cache = forward_propagation(inner_X, inner_params)
        inner_cost = compute_cost(inner_A2, inner_Y, inner_params)
        inner_grads = backward_propagation(inner_params, inner_cache, inner_X, inner_Y)
        inner_params = update_parameters(inner_params, inner_grads, inner_learning_rate=learning_rate)

        if print_cost:
            if (i+1) % 1000 == 0:
                print("第 ", i+1, " 次循环，成本为: " + str(inner_cost))

    return inner_params


# 测试nn_model
# print("====================测试nn_model====================")
# X_assess, Y_assess = nn_model_test_case()
# parameters = nn_model(X_assess, Y_assess, 4, num_iteration=10000, learning_rate=1.2, print_cost=True)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))
