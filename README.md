# machine_learn

本仓库收录了机器学习基础算法的实践项目，涵盖回归与分类两大任务。每个子项目均包含完整的数据处理、模型实现、评估分析与实验文档，适合作为机器学习入门学习的参考案例。

---

## 项目结构

```
machine_learn/
├── README.md                           # 本文件
├── 线性回归/                           # 实验一：线性回归
│   ├── src/
│   │   ├── linear.py                   # 主程序（梯度下降 + 正规方程）
│   │   └── housing.csv                 # California Housing 数据集
│   ├── dataset/
│   │   └── housing_linear.csv          # 备用数据集
│   └── 黄科海_04230927.docx            # 实验报告
└── 逻辑回归与朴素贝叶斯分类/           # 实验二：分类算法
    ├── README.md                       # 子项目说明
    ├── data/
    │   ├── iris.data                   # 鸢尾花数据集
    │   ├── iris.names                  # 数据集说明
    │   ├── bezdekIris.data             # 备用数据集
    │   └── Index
    ├── src/
    │   └── test.py                     # 主程序（逻辑回归 + 朴素贝叶斯）
    ├── docs/
    │   ├── 实验二 逻辑回归与朴素贝叶斯分类.pdf
    │   └── 黄科海_04230927.docx        # 实验报告
    └── dataset/                        # 数据副本
        ├── iris.data
        ├── iris.names
        └── bezdekIris.data
```

---

## 子项目一：线性回归

### 简介
基于 **California Housing** 数据集，使用 Python 从零实现线性回归模型，对比两种参数求解方法：
- **梯度下降法（Gradient Descent）**
- **正规方程法（Normal Equation）**

### 核心内容
- 数据预处理：独热编码（`ocean_proximity`）、缺失值处理、标准化
- 模型训练：手动实现损失函数、梯度下降迭代
- 模型评估：计算训练集与测试集的 **R² 分数**
- 特征分析：输出各特征与房价的相关系数
- 可视化：绘制梯度下降的损失曲线

### 运行方式
```bash
cd "线性回归/src"
python linear.py
```

---

## 子项目二：逻辑回归与朴素贝叶斯分类

### 简介
基于经典的 **Iris（鸢尾花）数据集**，实现并对比两种机器学习分类算法：
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

## 环境依赖

- Python 3.x
- NumPy
- Pandas
- Matplotlib（线性回归项目使用）
- scikit-learn（仅用于 `train_test_split`）

安装依赖：
```bash
pip install numpy pandas matplotlib scikit-learn
```

---

## 作者

黄科海（学号：04230927）
