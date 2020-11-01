import  numpy as np
import  pandas as pd
import random
#
from math import sqrt
from numpy import concatenate
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


x=np.random.randn(10,1)

def moving_window(x, length):
    return [x[i: i + length] for i in range(0, (len(x)+1)-length, length)]

x_ = moving_window(x, 3)
x_ = np.asarray(x_)
print(x_)
