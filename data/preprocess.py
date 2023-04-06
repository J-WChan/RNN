import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

__all__ = ['modifying']

def modifying(df:pd.DataFrame) :
    print('modifying')
    scaler = StandardScaler()
    df[['Price']] = scaler.fit_transform(df)
    data = df['Price'].values.tolist()

    batch_size = 3
    list_x, list_y = [], []

    for i in range(len(data)-batch_size) :
        temp_x = data[i:i+batch_size]
        temp_y = data[i+batch_size]
        list_x.append(temp_x)
        list_y.append(temp_y)

    array_x = np.array(list_x)
    array_y = np.array(list_y)

    train_x, test_x, train_y, test_y = train_test_split(array_x,array_y,test_size=0.2)

    train_x = train_x.reshape(train_x.shape[0],1,train_x.shape[1])
    test_x = test_x.reshape(test_x.shape[0],1,test_x.shape[1])

    return train_x, train_y,test_x, test_y