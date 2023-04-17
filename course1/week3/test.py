import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model
from course1.week3.utils.planar_utils import plot_decision_boundary, load_planar_dataset

# 载入数据集
X, Y = load_planar_dataset()
# 查看数据集分布(图)
plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral)
# 解除注释查看图像
# plt.show()

# 查看数据集的维度
shape_X = X.shape
shape_Y = Y.shape
m = Y.shape[1]
print("X的维度是：" + str(shape_X))
print("Y的维度是：" + str(shape_Y))
print("数据集大小：" + str(m))

# 使用sklearn里的逻辑回归，可以发现效果并不好
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.T)
# 画出边界
plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression")
LR_predictions = clf.predict(X.T)
# 计算准确率
corr_count = np.dot(Y, LR_predictions) + np.dot(1 - Y, 1 - LR_predictions)
corr_rate = corr_count / Y.size
print("准确率是: " + str(corr_rate[0] * 100) + "%")
plt.show()
