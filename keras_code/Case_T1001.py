# T1001
import pandas as pd
import numpy as np
import daetime
import maplotlib.pyplot as plt
from keras import models,layers,optimizers
import os
from keras.utils import to_categorical

newData1=pd.read_csv("./SimpleVersionData.csv")
newpdData1.loc[:,"Date"]=pd.to_datetime(newpdData1.loc[:,"Date"])
newpdData1=newpdData1.set_index("Date",drop=True)
data=np.array(newpdData1)

def generator(data,lookback,delay,min_index,max_index,shuffle=False,batch_size=32,step=1):
    if max_index is None:
        max_index=len(data)-delay-1
    i=min_index+lookback
    while 1:
        if shuffle:
            rows=np.random.randint(min_index+lookback,max_index,size=batch_size)
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
        
lookback=20
step=1
delay=1
batch_size=1024
trainnum=int(len(data)*0.8)
valnum=int(len(data)*0.2)

train_gen=generator(data,lookback=lookback,delay=delay,min_index=0,max_index=trainnum,step=step,batch_size=batch_size)
val_gen=generator(data,lookback=lookback,delay=delay,min_index=trainnum,max_index=None,step=step,batch_size=batch_size)


val_steps=(valnum-lookback)//batch_size
train_steps=trainnum//batch_size

model1=models.Sequential()
model1.add(layers.Dense(512,activation="relu",input_shape=(None,data.shape[-1])))
model1.add(layers.Dropout(0.5))
model1.add(layers.LSTM(32,dropout=0.5,return_sequences=True))
model1.add(layers.LSTM(64,dropout=0.5))
model1.add(layers.Dense(512,activation="relu"))
model1.add(layers.Dropout(0.5))
model1add(layers.Dense(8,activation="softmax"))
model1.summary()

model1.compile(optimizer=optimizers.RMSprop(lr=0.0001),loss="categorical_crossentropy",metrics=["acc"])
history1=model1.fit_generator(train_gen,steps_per_epoch=train_steps,epochs=200,validation_data=val_gen,validation_steps=val_steps)

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

model1.save("T1001.h5")
