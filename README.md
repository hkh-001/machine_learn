# machine_learn

本仓库收录了机器学习基础算法的实践项目，涵盖**回归、分类与聚类**三大任务。每个实验均包含完整的数据处理、模型实现、结果评估与可视化，适合作为机器学习入门学习的参考案例。

> **隐私说明**：为了保护个人隐私，所有 `.doc` / `.docx` 实验报告文档已通过 `.gitignore` 配置忽略，不会出现在 Git 提交记录中。

---

## 项目结构

```
machine_learn/
├── README.md                                 # 本文件（项目总览）
│
├── 线性回归/                                 # 实验一：线性回归
│   ├── src/
│   │   ├── linear.py                         # 主程序（梯度下降 + 正规方程）
│   │   └── housing.csv                       # California Housing 数据集
│   ├── dataset/
│   │   └── housing_linear.csv                # 备用数据集
│   └── 黄科海_04230927.docx                  # 实验报告（已 gitignore 忽略）
│
├── 逻辑回归与朴素贝叶斯分类/                 # 实验二：分类算法
│   ├── README.md                             # 子项目说明
│   ├── src/
│   │   └── test.py                           # 主程序（逻辑回归 + 朴素贝叶斯）
│   ├── data/
│   │   ├── iris.data                         # 鸢尾花数据集
│   │   ├── iris.names                        # 数据集说明
│   │   ├── bezdekIris.data                   # 备用数据集
│   │   └── Index
│   ├── dataset/                              # 数据副本
│   ├── docs/
│   │   ├── 实验二 逻辑回归与朴素贝叶斯分类.pdf
│   │   └── 黄科海_04230927.docx              # 实验报告（已 gitignore 忽略）
│   └── 实验二 逻辑回归与朴素贝叶斯分类.pdf   # 实验指导书
│
└── 聚类分析/                                 # 实验三：聚类算法
    ├── README.md                             # 子项目说明
    ├── .gitignore
    ├── src/
    │   └── code.py                           # K-Means + GMM-EM 实现
    ├── data/
    │   └── iris.data                         # 鸢尾花数据集
    ├── figures/
    │   ├── Figure_1.png                      # K-Means 聚类结果
    │   ├── Figure_2.png                      # K-Means 真实标签对比
    │   ├── Figure_3.png                      # GMM-EM 聚类结果
    │   └── Figure_4.png                      # GMM-EM 真实标签对比
    ├── docs/
    │   └── 实验三 聚类分析.pdf               # 实验指导书
    └── 黄科海_04230927.docx                  # 实验报告（已 gitignore 忽略）
```

---

## 实验一：线性回归

基于 **California Housing** 数据集，使用 Python 从零实现线性回归模型，对比两种参数求解方法：
- **梯度下降法（Gradient Descent）**
- **正规方程法（Normal Equation）**

### 核心内容
- 数据预处理：独热编码（`ocean_proximity`）、缺失值处理、标准化
- 模型训练：手动实现损失函数与梯度下降迭代
- 模型评估：计算训练集与测试集的 **R² 分数**
- 特征分析：输出各特征与房价的相关系数
- 可视化：绘制梯度下降的损失曲线

### 运行方式
```bash
cd "线性回归/src"
python linear.py
```

---

## 实验二：逻辑回归与朴素贝叶斯分类

基于经典的 **Iris（鸢尾花）数据集**，实现并对比两种分类算法：
- **逻辑回归（Logistic Regression）**：梯度上升求解
- **朴素贝叶斯（Naive Bayes）**：基于高斯分布假设

将三分类问题转化为 **二分类问题**（`Iris-virginica` vs 其他）。

### 核心内容
- 数据预处理：类别标签转换、训练/测试集划分（7:3）
- 逻辑回归：Sigmoid 函数、手动实现梯度上升、交叉熵损失
- 朴素贝叶斯：计算先验概率与条件概率，基于贝叶斯定理预测
- 模型评估：准确率、精确率、召回率、F1-score、混淆矩阵（TP/FP/FN/TN）

### 运行方式
```bash
cd "逻辑回归与朴素贝叶斯分类/src"
python test.py
```

---

## 实验三：聚类分析

基于 **Iris（鸢尾花）数据集**，使用 `sepal length` 与 `petal length` 两维特征，实现两种无监督聚类算法：
- **K-Means 聚类**：手动实现迭代聚类过程
- **GMM-EM（高斯混合模型）**：基于 EM 算法进行概率聚类

### 核心内容
- 随机初始化与迭代收敛
- 聚类结果可视化与真实标签对比
- 通过标签映射计算聚类准确率

### 运行方式
```bash
cd "聚类分析/src"
python code.py
```

---

## 环境依赖

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- SciPy（实验三使用）

安装依赖：
```bash
pip install numpy pandas matplotlib scikit-learn scipy
```

---

## 许可证

本项目仅供学习参考使用。
