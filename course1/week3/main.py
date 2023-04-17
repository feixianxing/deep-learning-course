from model import nn_model
from course1.week3.utils.planar_utils import plot_decision_boundary, load_planar_dataset
import matplotlib.pyplot as plt
from predict import predict
import numpy as np

# 载入数据集
X, Y = load_planar_dataset()
# 正式运行
n_h = 4
parameters = nn_model(X, Y, inner_n_h=n_h, num_iteration=10000, learning_rate=0.5, print_cost=True)
# 绘制边界
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(n_h))

predictions = predict(parameters, X)
corr_count = np.squeeze(
    np.dot(Y, predictions.T) + np.dot(1-Y, 1-predictions.T)
)
total_count = Y.size
corr_rate = corr_count / total_count
print("准确率为: " + str(corr_rate*100) + "%")
plt.show()
