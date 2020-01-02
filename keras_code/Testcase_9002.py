#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 导入相关的库文件
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from keras import models,layers,optimizers
import os
from keras.utils import to_categorical

# 导入简化后的原始数据
newData1=pd.read_csv("./SimpleVersionData.csv")
newData1.loc[:,"Date"]=pd.to_datetime(newData1.loc[:,"Date"])
newData1=newData1.set_index("Date",drop=True)

#转换数据为numpy格式
data=np.array(newData1)

#定义数据生成器
#data：浮点数组成的原始数组
#lookback: 输入包括的过去的时间步长
#delay： 目标在未来多少个时间步之后
#min_index和max_index:data数组中的索引，用于界定需要抽取哪些时间步，
#有助于保存一部分数据用于验证，一部分用户测试
# shuffle: 是打乱样本，还是顺序抽取样本
# batch_size: 每个批量的样本数
# step： 数据采样的周期
def generator(data,
              lookback,
              delay,
              min_index,
              max_index,
              shuffle=False,
              batch_size=32,
              step=1):
    if max_index is None:
        max_index=len(data)-delay-1
    i=min_index+lookback
    while 1:
        if shuffle:
            rows=np.random.randint(min_index+lookback,
                                   max_index,size=batch_size)
        else:
            if i+batch_size>=max_index:
                i=min_index+lookback
            rows=np.arange(i,min(i+batch_size,max_index))
            i+=len(rows)
        samples=np.zeros((len(rows),lookback//step,data.shape[-1]))
        targets=np.zeros((len(rows),))
        for j,row in enumerate(rows):
            indices=range(rows[j]-lookback,rows[j],step)
            samples[j]=data[indices]
            targets[j]=data[rows[j]+delay][-1]
        yield samples,to_categorical(targets,num_classes=8)

# 定义超参数
lookback=20
step=1
delay=1
batch_size=1024
trainnum=int(len(data)*0.8)
valnum=int(len(data)*0.2)

#实例化生成器
train_gen=generator(data,
                    lookback=lookback,
                    delay=delay,
                    min_index=0,
                    max_index=trainnum,
                    step=step,
                    batch_size=batch_size)
val_gen=generator(data,
                  lookback=lookback,
                  delay=delay,
                  min_index=trainnum,
                  max_index=None,
                  step=step,
                  batch_size=batch_size)

# 计算val_steps，这是计算需要从val_gen中抽取多少次
val_steps=(valnum-lookback)//batch_size
# 查看训练集需要抽取的次数
train_steps=trainnum//batch_size

from keras.callbacks import EarlyStopping
early_stopping=EarlyStopping(monitor="val_loss",patience=30)
from keras.callbacks import ReduceLROnPlateau
lr_reduce=ReduceLROnPlateau(monitor="val_loss,factor=0.1,patience=10")

#创建模型
model1=models.Sequential()
model1.add(layers.Dense(512,activation="tanh",input_shape=(None,data.shape[-1])))
model1.add(layers.Dropout(0.5))
model1.add(layers.BatchNormalization())
model1.add(layers.LSTM(32,activation="tanh",dropout=0.5,return_sequences=True))
model1.add(layers.BatchNormalization())
model1.add(layers.LSTM(32,activation="tanh",dropout=0.5,return_sequences=True))
model1.add(layers.BatchNormalization())
model1.add(layers.LSTM(32,activation="tanh",dropout=0.5,return_sequences=True))
model1.add(layers.BatchNormalization())
model1.add(layers.LSTM(32,activation="tanh",dropout=0.5))
model1.add(layers.BatchNormalization())
model1.add(layers.Dense(512,activation="tanh"))
model1.add(layers.Dropout(0.5))
model1.add(layers.BatchNormalization())
model1.add(layers.Dense(8,activation="softmax"))
model1.summary()

model1.compile(optimizer=optimizers.RMSprop(),
               loss="categorical_crossentropy",
               metrics=["acc"])
history1=model1.fit_generator(train_gen,
                              steps_per_epoch=train_steps,
                              epochs=200,
                              validation_data=val_gen,
                              validation_steps=val_steps,
                              callbacks=[lr_reduce]
                              )

# 绘制结果
loss=history1.history["loss"]
val_loss=history1.history["val_loss"]
acc=history1.history["acc"]
val_acc=history1.history["val_acc"]
epochs=range(1,len(loss)+1)
plt.figure(figsize=(10,5))
plt.plot(epochs,acc,"b",c="black",label="acc")
plt.plot(epochs,val_acc,"b",c="green",label="val_acc")
plt.title("Train and validation curve")
plt.legend()

#保存模型
model1.save("T9002.h5")

