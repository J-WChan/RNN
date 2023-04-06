from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM
import numpy as np

__all__ = ['model']

def model(train_x,train_y,test_x) : 
    print('model')
    model = Sequential()
    model.add(LSTM(units=50,activation='tanh',input_shape=(1,3),return_sequences=True))
    model.add(LSTM(units=40,activation='tanh',return_sequences=True))
    model.add(LSTM(units=30,activation='tanh'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss ='mean_squared_error')
    model.fit(train_x,train_y, epochs=10)
    pred_y = model.predict(test_x)
    return pred_y