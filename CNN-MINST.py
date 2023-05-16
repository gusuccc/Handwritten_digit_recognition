# matplotlib 用于绘图
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['simHei']#显示中文标签

# 处理数据的库
import numpy as np
import sklearn
import pandas as pd
# 系统库
import os
import sys
import time
# TensorFlow的库
import tensorflow as tf
from tensorflow import keras

#导入MINST数据集
mnist = tf.keras.datasets.mnist
(train_x, train_y), (test_x,test_y) = mnist.load_data()
#print(train_x.shape)

#属性归一化
X_train,X_test = tf.cast(train_x, dtype=tf.float32)/255.0,tf.cast(test_x,dtype=tf.float32)/255.0
y_train,y_test = tf.cast(train_y, dtype=tf.int32),tf.cast(test_y,dtype=tf.int32)
#增加通道维度
X_train = train_x.reshape (60000,28,28,1)
X_test = test_x.reshape(10000,28,28,1)
#print(X_train.shape)

#建立模型
model=tf.keras.Sequential([
#特征层
#unit 1——卷积层1&池化层1 卷积核16，大小3*3，输入28*28*1，池化层2*2
tf.keras.layers.Conv2D(16,kernel_size=(3,3),padding="same", activation=tf.nn.relu,input_shape=(28,28,1)) ,
tf.keras.layers.MaxPool2D(pool_size=(2,2)),

# unit 2——卷积层2&池化层2 卷积核32，大小3*3，池化层2*2
tf.keras.layers.Conv2D(32,kernel_size=(3,3),padding="same", activation=tf.nn.relu),
tf.keras.layers.MaxPool2D(pool_size=(2,2)),

#分类识别层
#unit 3——Flatten张量展开层
tf.keras.layers.Flatten(),

# unit 4——隐含层&输出层，隐含层128结点，输出层10结点
tf.keras.layers.Dense(128,activation="relu"),
tf.keras.layers. Dense(10, activation="softmax")
])
#model.summary()#查看模型摘要

#配置训练方法
model.compile (optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['sparse_categorical_accuracy'])
#训练
history = model.fit(X_train,y_train, batch_size=64, epochs=5, validation_split=0.2)

#评估模型
model.evaluate(X_test,y_test,verbose=2)
#保存训练日志文件
pd.DataFrame(history.history).to_csv("training_log.csv" , index=False)
#绘制训练曲线
graph = pd.read_csv('training_log.csv')
graph. plot (figsize = (8,5))
plt.grid(1)
plt.xlim(0,4)
plt.ylim(0, 1)
plt.show()

#预测数据
plt.figure()
for i in range(10):
    num= np. random. randint (1,10000)

    plt.subplot (2,5,i+1)
    plt. axis ('off')
    plt.imshow(test_x[num] , cmap='gray')
    demo = tf.reshape(X_test [num],(1,28,28,1))
    y_pred = np. argmax (model.predict(demo))
    plt.title('标签值:'+str(test_y[num])+'\n预测值: '+str(y_pred))
plt.show ()

'''
保存和加载模型
model.save_weights("minst.h5")
model.load_weights("minst.h5")
'''
