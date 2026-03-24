import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


data = pd.read_csv(r"C:\Users\huang\Desktop\machine_learn\线性回归\src\housing.csv")

print("原始数据前5行：")
print(data.head())
print("\n数据信息：")
print(data.info())
print("\nocean_proximity类别统计：")
print(data["ocean_proximity"].value_counts())


# =========================
# 2. 数据预处理
# =========================

# 2.1 one-hot 编码
data = pd.get_dummies(data, columns=["ocean_proximity"], prefix="ocean", dtype=int)
print(data)
# 2.2 删除缺失值（total_bedrooms有缺失）
data = data.dropna(axis=0)

print("\n处理后的数据形状：", data.shape)

tmp = data.drop('median_house_value', axis=1)
y = data['median_house_value']
data_2 = pd.concat([tmp, y], axis=1)
print(data_2)

# 2.3 划分特征X和目标y
X_df = data.drop("median_house_value", axis=1)
y_df = data["median_house_value"]

# 2.4 划分训练集和测试集（70% / 30%）
X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(
    X_df, y_df, test_size=0.3, random_state=42
)

print("\n训练集特征形状：", X_train_df.shape)
print("测试集特征形状：", X_test_df.shape)

X = X_train_df
y = y_train_df
X_test = X_test_df
y_test = y_test_df

print("X:")
print(X)
print("y:")
print(y)
X = np.matrix(X.values)
y = np.matrix(y.values)
X_test = np.matrix(X_test.values)
y_test = np.matrix(y_test.values)

print("X矩阵：")
print(X)
print("y矩阵：")
print(y)

# =========================
# 3. 特征缩放（标准化）
#    注意：测试集必须用训练集的均值和标准差
# =========================
train_mean = X_train_df.mean()
train_std = X_train_df.std()

# 防止某一列标准差为0
train_std = train_std.replace(0, 1)

X_train_std = (X_train_df - train_mean) / train_std
X_test_std = (X_test_df - train_mean) / train_std

# 转为 numpy 数组
X_train = X_train_std.values
X_test = X_test_std.values
y_train = y_train_df.values.reshape(-1, 1)
y_test = y_test_df.values.reshape(-1, 1)

# 给 X 加偏置项 1
X_train = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]

print("\n加偏置项后：")
print("X_train shape =", X_train.shape)
print("X_test shape =", X_test.shape)
print("y_train shape =", y_train.shape)
print("y_test shape =", y_test.shape)


# =========================
# 4. 定义损失函数
# =========================
def compute_cost(X, y, theta):
    """
    计算线性回归损失函数:
    J(theta) = 1/(2m) * sum((X theta - y)^2)
    """
    m = len(y)
    errors = X @ theta - y
    cost = (1 / (2 * m)) * np.sum(errors ** 2)
    return cost


# =========================
# 5. 梯度下降
# =========================
def gradient_descent(X, y, theta, alpha, num_iters):
    """
    梯度下降法求解参数
    """
    m = len(y)
    cost_history = []

    for i in range(num_iters):
        gradient = (1 / m) * (X.T @ (X @ theta - y))
        theta = theta - alpha * gradient
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

    return theta, cost_history


# 初始化参数
theta_init = np.zeros((X_train.shape[1], 1))

# 超参数
alpha = 0.01
num_iters = 2000

# 训练
theta_gd, cost_history = gradient_descent(X_train, y_train, theta_init, alpha, num_iters)

print("\n梯度下降求得的参数 theta：")
print(theta_gd)


# =========================
# 6. 正规方程
# =========================
def normal_equation(X, y):
    """
    正规方程求解:
    theta = (X^T X)^(-1) X^T y
    使用伪逆 pinv 更稳定
    """
    theta = np.linalg.pinv(X.T @ X) @ X.T @ y
    return theta


theta_ne = normal_equation(X_train, y_train)

print("\n正规方程求得的参数 theta：")
print(theta_ne)


# =========================
# 7. 预测函数
# =========================
def predict(X, theta):
    return X @ theta


# =========================
# 8. R² 评价指标
# =========================
def r2_score_manual(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)   # 残差平方和
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # 总离差平方和
    r2 = 1 - ss_res / ss_tot
    return r2


# 训练集预测
y_train_pred_gd = predict(X_train, theta_gd)
y_train_pred_ne = predict(X_train, theta_ne)

# 测试集预测
y_test_pred_gd = predict(X_test, theta_gd)
y_test_pred_ne = predict(X_test, theta_ne)

# 计算 R²
r2_train_gd = r2_score_manual(y_train, y_train_pred_gd)
r2_test_gd = r2_score_manual(y_test, y_test_pred_gd)

r2_train_ne = r2_score_manual(y_train, y_train_pred_ne)
r2_test_ne = r2_score_manual(y_test, y_test_pred_ne)

print("\n========== 模型评估结果 ==========")
print("梯度下降法：")
print("训练集 R² =", r2_train_gd)
print("测试集 R² =", r2_test_gd)

print("\n正规方程法：")
print("训练集 R² =", r2_train_ne)
print("测试集 R² =", r2_test_ne)


# =========================
# 9. 绘制损失曲线
# =========================
plt.figure(figsize=(8, 5))
plt.plot(range(len(cost_history)), cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Gradient Descent Cost Curve")
plt.grid(True)
plt.show()


# =========================
# 10. 相关系数分析（可选，实验文档里有类似内容）
# =========================
corr = data.corr(numeric_only=True)
score = corr["median_house_value"].sort_values()

print("\n各特征与 median_house_value 的相关系数：")
print(score)