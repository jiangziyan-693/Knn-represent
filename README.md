# K-Nearest Neighbors (KNN) Classifier
该项目展示了使用鸢尾花数据集实现K-最近邻（KNN）分类器。分类器可以配置为使用单个`k`值或测试一系列`k`值以找到最佳值。

## 文件

- `config.yaml`：包含KNN分类器参数的配置文件。
- `main.py`：运行KNN分类器并可视化结果的主脚本。

## 配置文件 (`config.yaml`)

```yaml
n_neighbors_lowest: 1
n_neighbors_highest: 55
n_neighbors: 5
whether_k_test: false
```

- `n_neighbors_lowest`：要测试的最小`k`值。
- `n_neighbors_highest`：要测试的最大`k`值。
- `n_neighbors`：如果`whether_k_test`设置为`false`，则使用的`k`值。
- `whether_k_test`：布尔标志，确定是否测试一系列`k`值。

## 使用方法

1. 确保已安装所需的库：
    ```bash
    pip install numpy pandas matplotlib scikit-learn pyyaml
    ```

2. 修改`config.yaml`文件以设置所需的参数。

3. 运行`main.py`脚本：
    ```bash
    python main.py
    ```

## 可视化

该脚本提供了单个`k`值分类和`k`范围测试的可视化：
