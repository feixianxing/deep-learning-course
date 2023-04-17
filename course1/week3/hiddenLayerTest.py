from model import nn_model
from course1.week3.utils.planar_utils import plot_decision_boundary, load_planar_dataset
import matplotlib.pyplot as plt
from predict import predict
import numpy as np

# 载入数据集
X, Y = load_planar_dataset()

plt.figure(figsize=(24, 16))
# 隐藏层数量
hidden_layer_sizes = [1, 2, 3, 4, 5, 10, 20, 30, 50]

for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(3, 3, i+1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(X, Y, n_h, num_iteration=5000, learning_rate=1.2, print_cost=False)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
    print("隐藏层的节点数量： {}  ，准确率: {} %".format(n_h, accuracy))

plt.show()
