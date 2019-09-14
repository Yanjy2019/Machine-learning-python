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
