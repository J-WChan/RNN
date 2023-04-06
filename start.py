from data import loading
from data import modifying
from model import model
import setting
# import click
import matplotlib.pyplot as plt

def start():
    # print('file_name')
    df = loading(path=setting.path)
    train_x,train_y,test_x,test_y = modifying(df)
    pred_y = model(train_x,train_y,test_x)
    plt.plot(test_y[:100],label='real')
    plt.plot(pred_y[:100],label='pred')
    plt.legend()
    plt.show()
    