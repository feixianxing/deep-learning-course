import numpy as np
from steps import *


# 预测
def predict(inner_param, inner_X):
    """
    使用训练好的参数，为X中的每个示例预测
    :param inner_param: 包含参数的字典类型变量
    :param inner_X: 输入数据 (n_x, m)
    :return:
        predictions: 模型预测的向量(红色：0，蓝色：1)
    """
    A2, cache = forward_propagation(inner_X, inner_param)
    inner_predictions = np.round(A2)

    return inner_predictions


# 测试predict
print("====================测试predict====================")
parameters, X_assess = predict_test_case()
predictions = predict(parameters, X_assess)
print("预测的平均值 = " + str(np.mean(predictions)))
