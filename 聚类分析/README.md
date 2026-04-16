# 实验三：聚类分析

本项目实现了两种经典的聚类算法：**K-Means** 和 **GMM-EM（高斯混合模型）**，并在经典的 Iris 数据集上进行实验验证。

## 项目结构

```
.
├── data/               # 数据集
│   └── iris.data       # Iris 花卉数据集
├── src/                # 源代码
│   └── code.py         # K-Means + GMM-EM 算法实现
├── figures/            # 实验结果可视化图片
│   ├── Figure_1.png
│   ├── Figure_2.png
│   ├── Figure_3.png
│   └── Figure_4.png
├── docs/               # 实验文档
│   └── 实验三 聚类分析.pdf
├── README.md
└── .gitignore
```

## 算法说明

### K-Means 聚类
- 随机初始化 3 个聚类中心
- 迭代计算每个样本到中心点的距离并重新归类
- 更新聚类中心，直到收敛
- 实验使用 `sepal length` 和 `petal length` 两维特征

### GMM-EM 聚类
- 使用高斯混合模型对数据进行概率建模
- 通过 EM 算法（期望最大化）迭代估计模型参数
- 对初始值敏感，实验中对数据进行了 Min-Max 归一化

## 运行方式

```bash
cd src
python code.py
```

## 依赖

- Python 3.x
- numpy
- pandas
- matplotlib
- scikit-learn
- scipy

## 注意事项

- `.doc` / `.docx` 文件已被 `.gitignore` 忽略，不会提交到仓库中。
- `figures/` 中的图片为运行代码后生成的实验结果。
