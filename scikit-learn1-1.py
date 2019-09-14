#k近邻算法的整体调用流程

#1-1输入任意的自定义数据集来进行相关的验证
import numpy as np
import matplotlib.pyplot as plt #导入相应的数据可视化模块
raw_data_X=[[3.393533211,2.331273381],
            [3.110073483,1.781539638],
            [1.343808831,3.368360954],
            [3.582294042,4.679179110],
            [2.280362439,2.866990263],
            [7.423436942,4.696522875],
            [5.745051997,3.533989803],
            [9.172168622,2.511101045],
            [7.792783481,3.424088941],
            [7.939820817,0.791637231]]
raw_data_Y=[0,0,0,0,0,1,1,1,1,1]
print(raw_data_X)
print(raw_data_Y)
x_train=np.array(raw_data_X)
y_train=np.array(raw_data_Y)  #数据的预处理，需要将其先转换为矩阵，并且作为训练数据集
print(x_train)
print(y_train)
plt.figure(1)
plt.scatter(x_train[y_train==0,1],x_train[y_train==0,0],color="g")
plt.scatter(x_train[y_train==1,0],x_train[y_train==1,1],color="r")  #将其散点图输出
x=np.array([8.093607318,3.365731514])  #定义一个新的点，需要判断它到底属于哪一类数据类型
plt.scatter(x[0],x[1],color="b")  #在算点图上输出这个散点，看它在整体散点图的分布情况
#kNN机器算法的使用
from math import sqrt
distance=[]
for x_train in x_train:
    d=sqrt(np.sum((x_train-x)**2))
    distance.append(d)
print(distance)
d1=np.argsort(distance)  #输出distance排序的索引值
print(d1)
k=6
n_k=[y_train[(d1[i])] for i in range(0,k)]
print(n_k)
from collections import Counter   #导入Counter模块
c=Counter(n_k).most_common(1)[0][0]   #Counter模块用来输出一个列表中元素的个数，输出的形式为列表，其里面的元素为不同的元组
#另外的话对于Counter模块它有.most_common(x)可以输出统计数字出现最多的前x个元组，其中元组的key是其元素值，后面的值是出现次数
y_predict=c
print(y_predict)
plt.show()  #输出点的个数
#在scikitlearn中调用KNN算法的操作步骤
from sklearn.neighbors import KNeighborsClassifier
KNN_classifier=KNeighborsClassifier(n_neighbors=6)
raw_data_X=[[3.393533211,2.331273381],
            [3.110073483,1.781539638],
            [1.343808831,3.368360954],
            [3.582294042,4.679179110],
            [2.280362439,2.866990263],
            [7.423436942,4.696522875],
            [5.745051997,3.533989803],
            [9.172168622,2.511101045],
            [7.792783481,3.424088941],
            [7.939820817,0.791637231]]
raw_data_Y=[0,0,0,0,0,1,1,1,1,1]
print(raw_data_X)
print(raw_data_Y)
x_train=np.array(raw_data_X)
y_train=np.array(raw_data_Y)
print(x_train)
print(y_train)
KNN_classifier.fit(x_train,y_train)
print(x)
x=x.reshape(1,-1)
print(KNN_classifier.predict(x))
test_data1=[[3.93533211,2.33127381],
            [3.10073483,1.78159638],
            [1.34808831,3.36830954],
            [3.58294042,4.67919110],
            [2.28032439,2.86690263],
            [7.42343942,4.69652875],
            [5.74505997,3.53399803],
            [9.17216622,2.51101045],
            [7.79278481,3.42488941],
            [7.93982087,0.79637231]]
test_data=np.array(test_data1)
test_target=[0,0,0,0,1,1,0,0,0,0]
y_pred=KNN_classifier.predict(test_data)
from sklearn import metrics   #引入机器学习的验证模块
print(metrics.accuracy_score(y_true=test_target,y_pred=y_pred))  #输出整体预测结果的准确率，其中第三个参数normalize=False表示输出结果预测正确的个数
print(metrics.confusion_matrix(y_true=test_target,y_pred=y_pred)) #输出混淆矩阵，如果为对角阵，则表示预测结果是正确的，准确度越大


#1-2利用scikitlearn自带的iris数据集进行相关的训练
import numpy as np
import pandas as pd
#引入原始数据,进行数据的预处理
from sklearn.datasets import load_iris #导入iris原始数据集合
iris=load_iris()
print(iris)
print(len(iris["data"]))
from  sklearn.model_selection import train_test_split #引入数据训练与检验模块
train_data,test_data, train_target, test_target=train_test_split(iris.data,iris.target,test_size=0.1,random_state=1)
#建立数据的模型和相应的决策树结构
from sklearn.neighbors import KNeighborsClassifier
KNN_classifier=KNeighborsClassifier(n_neighbors=6)
KNN_classifier.fit(train_data,train_target)        #进行原始数据的训练
y_pred=KNN_classifier.predict(test_data)                  #进行数据集的测试

#数据验证
from sklearn import metrics   #引入机器学习的验证模块
print(metrics.accuracy_score(y_true=test_target,y_pred=y_pred))  #输出整体预测结果的准确率，其中第三个参数normalize=False表示输出结果预测正确的个数
print(metrics.confusion_matrix(y_true=test_target,y_pred=y_pred)) #输出混淆矩阵，如果为对角阵，则表示预测结果是正确的，准确度越大


#1-3利用scikitlearn自带的手写字体digits数据集进行相关的训练
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import matplotlib
digits=datasets.load_digits()   #导入手写字体数据集
print(digits.keys())
x=digits.data
print(x.shape)
y=digits.target
print(y.shape)
print(y[:100])
print(x[:10])
x1=x[666].reshape(8,8)
print(x1)
plt.imshow(x1,cmap=matplotlib.cm.binary)
plt.show()
print(y[666])
from  sklearn.model_selection import train_test_split #引入数据训练与检验模块
x_train,x_test, y_train, y_test=train_test_split(digits.data,digits.target,test_size=0.1,random_state=0)
#建立数据的模型和相应的KNNs算法结构
from sklearn.neighbors import KNeighborsClassifier
KNN=KNeighborsClassifier(n_neighbors=3)
KNN_classifier.fit(x_train,y_train)        #进行原始数据的训练
y_pred=KNN_classifier.predict(x_test)                  #进行数据集的测试
print(y_pred)
print(KNN_classifier.score(x_test,y_test))  #直接输出相应的准确度
#数据验证

from sklearn import metrics   #引入机器学习的验证模块
print(metrics.accuracy_score(y_true=y_test,y_pred=y_pred))  #输出整体预测结果的准确率，其中第三个参数normalize=False表示输出结果预测正确的个数
print(metrics.confusion_matrix(y_true=y_test,y_pred=y_pred)) #输出混淆矩阵，如果为对角阵，则表示预测结果是正确的，准确度越大
from  sklearn.model_selection import train_test_split #引入数据训练与检验模块
x_train,x_test, y_train, y_test=train_test_split(digits.data,digits.target,test_size=0.2,random_state=0)
#建立数据的模型和相应的KNNs算法结构

#对于KNN算法寻找最佳的超参数k的值以及另外一个超参数distances,以及在distance的情况下选择出最佳的超参数p的值的大小
best_method=""
best_score=0.0
best_k=0
s=[]
from sklearn.neighbors import KNeighborsClassifier
for method in ["uniform","distance"]:
    for k in range(1,11):
        KNN=KNeighborsClassifier(n_neighbors=k,weights=method)
        KNN.fit(x_train,y_train)        #进行原始数据的训练
        score=KNN.score(x_test,y_test) #直接输出相应的准确度
        s.append(score)
        if score>best_score:
            best_score=score
            best_k=k
            best_method=method
        #数据验证
print("best_method=",best_method)
print("best_k=",best_k)
print("best_score=",best_score)
plt.figure(2)
x=[i for i in range(1,21)]
plt.plot(x,s,"r")
plt.show()

best_p=0
best_score=0.0
best_k=0
s=[]
from sklearn.neighbors import KNeighborsClassifier
for k in range(1,11):
    for p in range(1,6):
        KNN=KNeighborsClassifier(n_neighbors=k,weights="distance",p=p)
        KNN.fit(x_train,y_train)        #进行原始数据的训练
        score=KNN.score(x_test,y_test) #直接输出相应的准确度
        s.append(score)
        if score>best_score:
            best_score=score
            best_k=k
            best_p=p
            #数据验证
print("best_p=",best_p)
print("best_k=",best_k)
print("best_score=",best_score)
plt.figure(2)
s1=[]
x=[i for i in range(1,6)]
for i in range(1,11):
    s1=s[(i*5-5):(5*i)]
    plt.plot(x,s1,label=i)
    plt.legend(loc=2)
plt.show()

#使用scikitlearn中的gridsearch来进行机器学习算法的超参数的最佳网格搜索方式
param_grid=[{
    "weights":["uniform"],
    "n_neighbors":[i for i in range(1,11)]
},
    {"weights":["distance"],
    "n_neighbors":[i for i in range(1,11)],
     "p":[i for i in range(1,6)]
    }
]       #定义机器学习算法的不同超参数组合，使用字典的方式，二对于具体的超参数采用列表的数据结构
knn_clf=KNeighborsClassifier()
from sklearn.model_selection import GridSearchCV
grid_search=GridSearchCV(knn_clf,param_grid,n_jobs=-1,verbose=2)
grid_search.fit(x_train,y_train)
print(grid_search.best_estimator_)
print(grid_search.best_params_)
print(grid_search.best_score_)


#Scaler数据归一化处理之后的KNN算法训练与实现
import numpy as np
from sklearn import datasets
iris=datasets.load_iris()
x=iris.data
y=iris.target
print(x[:10])
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=666)

#1-1对于x_train利用均值方差进行归一化处理
from sklearn.preprocessing import StandardScaler
standardscaler=StandardScaler()
standardscaler.fit(x_train)
print(standardscaler.mean_) #平均值向量
print(standardscaler.scale_) #标准差向量
print(standardscaler.transform(x_train))
x_train=standardscaler.transform(x_train)
print(x_train)
x_test_standard=standardscaler.transform(x_test)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
print(knn.score(x_test_standard,y_test))

#1-2对于x_train利用均值归一化进行归一化处理
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=666)
from sklearn.preprocessing import MinMaxScaler
standardscaler1=MinMaxScaler()
standardscaler1.fit(x_train)
x_train=standardscaler1.transform(x_train)
print(x_train)
x_test_standard1=standardscaler1.transform(x_test)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
print(x_test_standard1)
print(knn.score(x_test_standard1,y_test))









