import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

path = r"C:\Users\huang\Desktop\machine_learn\逻辑回归与朴素贝叶斯分类\iris.data"

columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
data = pd.read_csv(path, header=None, names=columns)

print("数据列名：", data.columns.tolist())
print("类别列唯一值：", data['species'].unique())
print("数据量为：", len(data))
print(data.head())
print(data)

print("空值数量如下：")
print(data.isnull().sum())

print("各列数据数值分布为")
print(data.describe())

data['species'] = data['species'].apply(lambda x: 1 if x == 'Iris-virginica' else 0)
print(data)

counts = data['species'].value_counts()
print("非 virginica 的数量：", counts[0])
print("virginica 的数量：", counts[1])

train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

print("原始数据量{}".format(len(data)))
print("训练集数据量{}".format(len(train_data)))
print("测试集数据量{}".format(len(test_data)))

col = train_data.shape[1]
X = train_data.iloc[:, 0:col - 1].values
y = train_data.iloc[:, col - 1:col].values
X_test = test_data.iloc[:, 0:col - 1].values
y_test = test_data.iloc[:, col - 1:col].values

print("X矩阵为")
print(X)
print("\ny矩阵为")
print(y)

m = X.shape[0]
n = X_test.shape[0]

y = y.T
y_test = y_test.T

X = np.c_[np.ones(m), X]
X_test = np.c_[np.ones(n), X_test]

theta = np.matrix([0, 0, 0, 0, 0]).T
print(f"theta的形状为{theta.shape}")

alpha = 0.01
iters = 1000

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, theta):
    m_2 = len(y)
    predictions = sigmoid(X @ theta)
    cost = -1 / m_2 * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    return cost

y = y.reshape(-1, 1)
print(f"y的形状为{y.shape}")
print(f"X的形状为{X.shape}")

def gradient_boost(theta, alpha):
    for i in range(iters):
        theta = theta + (alpha / m) * (X.T @ (y - sigmoid(X @ theta)))
    return theta

theta = gradient_boost(theta, alpha)
print(f"theta的形状为{theta.shape}")

pred = sigmoid(X_test @ theta)
print(f"X_test的形状为{X_test.shape}")
print("预测概率为")
print(pred.flatten())
pred_class = (pred >= 0.5).astype(int)
print("预测序列为")
print(pred_class.flatten())
print("真值序列为")
print(y_test.flatten())

correct_num = np.sum(pred_class.flatten() == y_test)
print("正确预测个数为 {} 个，总个数有 {} 个".format(correct_num, n))
rate = float(correct_num) / n
print("正确率为 {}".format(rate))

TP = np.sum((pred_class.flatten() == 1) & (y_test == 1))
FP = np.sum((pred_class.flatten() == 1) & (y_test == 0))
FN = np.sum((pred_class.flatten() == 0) & (y_test == 1))
TN = np.sum((pred_class.flatten() == 0) & (y_test == 0))

print("TP 为 {}".format(TP))
print("FP 为 {}".format(FP))
print("FN 为 {}".format(FN))
print("TN 为 {}".format(TN))

precision = TP / (TP + FP)
recall = TP / (TP + FN)
accuracy = (TP + TN) / (TP + TN + FP + FN)
f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1_score)
print("accuracy:", accuracy)
print("---"*20)

v_prior = np.mean(y)
nv_prior = 1 - v_prior

y = y.flatten()

mean_v = X[:, 1:][y == 1].mean(axis=0)
std_v = X[:, 1:][y == 1].std(axis=0)
mean_nv = X[:, 1:][y == 0].mean(axis=0)
std_nv = X[:, 1:][y == 0].std(axis=0)

def gaussian_prob(x, mean, std):
    return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-0.5 * ((x - mean) / std) ** 2)

def predict(X):
    predictions = []
    for x in X:
        prob_v = np.prod(gaussian_prob(x, mean_v, std_v)) * v_prior
        prob_nv = np.prod(gaussian_prob(x, mean_nv, std_nv)) * nv_prior
        predictions.append(1 if prob_v > prob_nv else 0)
    return np.array(predictions)

pred = predict(X_test[:, 1:])
y_test = y_test.flatten()

print("原始真值序列为")
print(y_test)
print("预测序列为")
print(pred)

correct_num = np.sum(pred == y_test)
print("正确预测个数为 {} 个，总个数有 {} 个".format(correct_num, n))
rate = float(correct_num) / n
print("正确率为 {}".format(rate))

TP = np.sum((pred == 1) & (y_test == 1))
FP = np.sum((pred == 1) & (y_test == 0))
FN = np.sum((pred == 0) & (y_test == 1))
TN = np.sum((pred == 0) & (y_test == 0))

print("TP 为 {}".format(TP))
print("FP 为 {}".format(FP))
print("FN 为 {}".format(FN))
print("TN 为 {}".format(TN))

precision = TP / (TP + FP)
recall = TP / (TP + FN)
accuracy = (TP + TN) / (TP + TN + FP + FN)
f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1_score)
print("accuracy:", accuracy)
