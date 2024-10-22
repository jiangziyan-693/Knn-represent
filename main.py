import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import yaml
with open('config.yaml', 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
whether_k_test = config.get('whether_k_test')
if not whether_k_test:
    #导入iris数据
    iris = load_iris()
    X=iris.data[:,:2] #只取前两列
    y=iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,random_state=42) #划分数据，random_state固定划分方式
    #导入模型
    from sklearn.neighbors import KNeighborsClassifier 
    #训练模型
    n_neighbors = config.get('n_neighbors')
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    #查看各项得分
    print("y_pred",y_pred)
    print("y_test",y_test)
    print("score on train set", knn.score(X_train, y_train))
    print("score on test set", knn.score(X_test, y_test))
    print("accuracy score", accuracy_score(y_test, y_pred))

    # 可视化

    # 自定义colormap
    def colormap():
        return mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#FFC0CB','#00BFFF', '#1E90FF'], 256)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    axes=[x_min, x_max, y_min, y_max]
    xp=np.linspace(axes[0], axes[1], 500) #均匀500的横坐标
    yp=np.linspace(axes[2], axes[3],500) #均匀500个纵坐标
    xx, yy=np.meshgrid(xp, yp) #生成500X500网格点
    xy=np.c_[xx.ravel(), yy.ravel()] #按行拼接，规范成坐标点的格式
    y_pred = knn.predict(xy).reshape(xx.shape) #训练之后平铺

    # 可视化方法一
    plt.figure(figsize=(15,5),dpi=100)
    plt.subplot(1,2,1)
    plt.contourf(xx, yy, y_pred, alpha=0.3, cmap=colormap())
    #画三种类型的点
    p1=plt.scatter(X[y==0,0], X[y==0, 1], color='blue',marker='^')
    p2=plt.scatter(X[y==1,0], X[y==1, 1], color='green', marker='o')
    p3=plt.scatter(X[y==2,0], X[y==2, 1], color='red',marker='*')
    #设置注释
    plt.legend([p1, p2, p3], iris['target_names'], loc='upper right',fontsize='large')
    #设置标题
    plt.title(f"3-Class classification (k = {n_neighbors})", fontdict={'fontsize':15} )
    plt.show()
else:
    k_low = config.get('n_neighbors_lowest')
    k_high = config.get('n_neighbors_highest')
    iris = load_iris()
    X=iris.data[:,:2] #只取前两列
    y=iris.target
    k = k_low  
    x = []
    result = []
    while k <= k_high:
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,random_state=42) #划分数据，random_state固定划分方式
        #导入模型
        from sklearn.neighbors import KNeighborsClassifier 
        #训练模型
        n_neighbors = k
      
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        #查看各项得分
        print("y_pred",y_pred)
        print("y_test",y_test)
        print("score on train set", knn.score(X_train, y_train))
        print("score on test set", knn.score(X_test, y_test))
        print("accuracy score", accuracy_score(y_test, y_pred))
        result.append(accuracy_score(y_test, y_pred))
        x.append(k)
        k += 1
    
    # 创建折线图
    plt.plot(x, result, marker='o')
    # 添加标题和标签
    plt.title('k_test')
    plt.xlabel('k')
    plt.ylabel('accuracy')
    
    # 多项式回归
    x = np.array(x)
    x = x.reshape(-1, 1)
    degree = 2
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(x, result)
    y_pred = model.predict(x)
    plt.plot(x, y_pred, color='red', label=f'Polynomial Degree {degree}')
    # 显示网格
    plt.grid(True)
    # 添加图例
    plt.legend()
    # 显示图形
    plt.show()
