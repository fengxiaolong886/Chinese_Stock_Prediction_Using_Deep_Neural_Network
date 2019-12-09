
##bidirection
model1.add(layers.Bidirectional(layers.LSTM(128,activation="relu",return_sequences=True)))
##fit_generator callback
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
early_stopping=EarlyStopping(monitor="val_loss",patience=30)
lr_reduce=ReduceLROnPlateau(monitor="val_loss,factor=0.1,patience=10")
callbacks=[lr_reduce,early_stopping]


#cnn
model1.add(layers.Conv1D(filters=32,kernel_size=5,activation="relu"))
model1.add(layers.MaxPooling1D(5))
model1.add(layers.Dropout(0.5))

#GRU
model1.add(layers.GRU(32,activation="tanh",dropout=0.5))

#SimpleRNN
model1.add(layers.SimpleRNN(32,activation="relu",dropout=0.5))