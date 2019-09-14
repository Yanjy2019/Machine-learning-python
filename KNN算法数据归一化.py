import numpy as np
import matplotlib.pyplot as plt
x=np.random.randint(1,100,(50,2))
print(x)
x=np.array(x,dtype=float)
print(x)
x[:,0]=(x[:,0]-np.min(x[:,0]))/(np.max(x[:,0])-np.min(x[:,0]))
x[:,1]=(x[:,1]-np.min(x[:,1]))/(np.max(x[:,1])-np.min(x[:,1]))   #均值归一化处理实现
print(x)
plt.figure()
plt.scatter(x[:,0],x[:,1],color="r")
print(np.mean(x[:,0]))
print(np.std(x[:,0]))
print(np.mean(x[:,1]))
print(np.std(x[:,1]))
x[:,0]=(x[:,0]-np.mean(x[:,0]))/(np.std(x[:,0]))
x[:,1]=(x[:,1]-np.mean(x[:,1]))/(np.std(x[:,1]))             #均值方差归一化处理方式
print(x)
plt.scatter(x[:,0],x[:,1],color="g")
plt.show()
print(np.mean(x[:,0]))
print(np.std(x[:,0]))
print(np.mean(x[:,1]))
print(np.std(x[:,1]))