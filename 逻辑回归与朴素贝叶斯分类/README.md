# 逻辑回归与朴素贝叶斯分类

本项目使用经典的 Iris（鸢尾花）数据集，实现并对比了两种经典的机器学习分类算法：逻辑回归（Logistic Regression）和朴素贝叶斯（Naive Bayes）。

## 项目结构

```
.
├── data/                    # 数据集
│   ├── iris.data           # 鸢尾花数据集
│   ├── iris.names          # 数据集说明
│   ├── bezdekIris.data     # 备用数据集
│   └── Index               # 索引文件
├── src/                     # 源代码
│   └── test.py             # 主程序
├── docs/                    # 文档
│   ├── 实验二 逻辑回归与朴素贝叶斯分类.pdf
│   └── 黄科海_04230927.docx
└── README.md               # 项目说明
```

## 算法说明

### 1. 逻辑回归 (Logistic Regression)
- 使用梯度上升算法训练模型
- 学习率：0.01
- 迭代次数：1000
- 特征：花萼长度/宽度、花瓣长度/宽度（共4个特征）
- 二分类：Virginica vs 其他

### 2. 朴素贝叶斯 (Naive Bayes)
- 使用高斯分布假设
- 计算每个类别的先验概率和条件概率
- 基于贝叶斯定理进行预测

## 运行环境

- Python 3.x
- pandas
- numpy
- scikit-learn

## 使用方法

```bash
cd src
python test.py
```

## 输出指标

- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1-score
- 混淆矩阵 (TP/FP/FN/TN)

## 数据集

使用 UCI 机器学习仓库的 Iris 数据集，包含 150 个样本，3 个类别（本项目转为二分类问题）。

## 版本历史

- **v1.1.0**: 重构项目结构，添加 data/src 目录，纳入数据集，移除隐私文档
- **v1.0.0**: 初始版本，实现逻辑回归和朴素贝叶斯分类
