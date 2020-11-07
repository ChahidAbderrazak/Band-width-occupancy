"""
Band occupancy using DNN, RL.
Author: Abderrazak Chahid  |  abderrazak-chahid.com | abderrazak.chahid@gmail.com
"""




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

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from lib.Shared_Functions import *


####################################################################################
def load_bin_file(filename, max_lines=-1):
    print('-> Reading the bin file ', filename)
    file = open(filename, "rb")

    byte = file.read(1)
    cnt=1

    channels=[]
    while byte:

        # print(bin(byte))
        byte = file.read(1)

        if len(byte) > 0:
            # print(byte[0])
            if len(byte) > 1:
                print('byte of different size', len(byte))
                break

            channels.append(int(byte[0]))
            cnt=cnt+1

            if max_lines>0 and cnt > max_lines:
                break

    freq=np.asarray(channels)
    # data = pd.DataFrame(data=freq,  columns=["freq"])

    data = pd.DataFrame({'x': channels})

    return data

def normalize_shuffle(data, norm=0):
    data_=data; data_min=0; data_max=1;
    values = data_.values
    values = values.astype('float32')

    #%% normalize features
    if norm==1:
        scaler = MinMaxScaler(feature_range=(0, 1))
        values_normlized = scaler.fit_transform(values)

        # naorlaizin [0,1]
        data_min=values.min()
        values=values-data_min
        data_max=values.max()
        values_normlized=values/data_max
        data_ = DataFrame(values_normlized)
        print(data_)
        # input('flag')

    return data_, data_min, data_max

def sliding_windows_series(data0, windows_size=6, label_col = 'y'):
    data = pd.DataFrame(data0.copy())
    data.columns = [label_col]

    # add the lag of the target variable from 6 steps back up to 24
    for i in range(1, windows_size+1):
        data['var(t-{})'.format(i)] = data[label_col].shift(i)

    # data['time'] = data.index
    data = data.dropna()
    return data

def timeseries_train_test_deploy_split(X, y, deploy_size=0.4, test_size=0.4):
    """Perform train-test split with respect to time series structure."""
    deploy_index = int(len(X) * deploy_size)
    train_index = int((len(X)- deploy_index)* (1 - test_size))
    test_index = len(X)-train_index-deploy_index

    X_train = X[:train_index]
    X_test = X[train_index:train_index+test_index]
    X_deploy = X[train_index+test_index:]

    y_train = y[:train_index]
    y_test = y[train_index:train_index+test_index]
    y_deploy = y[train_index+test_index:]


    return X_train, X_test, X_deploy, y_train, y_test, y_deploy

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def run_ML_models(X, y, data_min=0, data_max=1, deploy_size=0.4, test_size=0.4, plot_intervals=False):
    """
    - Plots modelled vs original values.
    - Prediction intervals (95% confidence interval).
    - Anomalies (points that resides outside the confidence interval).
    """

    X_train, X_test, X_deploy, y_train, y_test, y_deploy = timeseries_train_test_deploy_split(X, y, deploy_size=deploy_size, test_size=test_size)

    # print('X_train=',X_train)

    # we are using random forest here, feel free to swap this out
    # with your favorite regression model
    model = RandomForestRegressor(max_depth=6, n_estimators=50)
    model.fit(X_train, y_train)

    if plot_intervals:
        timeseries_cv = TimeSeriesSplit(n_splits=5)
        cv = cross_val_score(model, X_train, y_train,
                         cv=timeseries_cv, scoring='neg_mean_absolute_error')

    #Prediction
    prediction = model.predict(X_deploy)
    MAE_deploy=plot_prediction(y_deploy, prediction, plot_intervals, cv, filename='predict_deploy')

    # predict all the data
    prediction = model.predict(X)
    MAE_all=plot_prediction(y, prediction, plot_intervals, cv, filename='predict_all')


    print('\n #################   RESTULTS   ##################')
    print('\n # MAE depolement = ', MAE_deploy)
    print('\n # MAE All  = ', MAE_all)
    print('\n #################################################')

    return model

def plot_prediction(ytrue, prediction, plot_intervals=False, cv=0, filename='results'):


    plt.figure(figsize=(15, 7))
    x = range(prediction.size)

    plt.plot(x, prediction, label='prediction', linewidth=2.0)
    plt.plot(x, ytrue, label='actual', linewidth=2.0)
    if plot_intervals:

        mae = -1 * cv.mean()
        deviation = cv.std()



    else:
        mae = 0
        deviation = 5


    scale = 1.96    # hard-coded to be 95% confidence interval

    margin_error = mae + scale * deviation
    lower = prediction - margin_error
    upper = prediction + margin_error

    fill_alpha = 0.2
    fill_color = '#66C2D7'
    plt.fill_between(x, lower, upper, color=fill_color, alpha=fill_alpha, label='95% Confidence Interval (CI)')


    anomalies = np.array([np.nan] * len(ytrue))
    anomalies = np.array([np.nan] * len(ytrue))
    anomalies[ytrue < lower] = ytrue[ytrue < lower]
    anomalies[ytrue > upper] = ytrue[ytrue > upper]
    plt.plot(anomalies, 'o', markersize=7, label='Anomalies outside the CI')

    MAE = mean_absolute_percentage_error(prediction, ytrue)
    plt.title('RF model [memory length='+str(windows_size)+']:  MAE=  '+ '{0:.2f}%'.format(MAE))
    plt.legend(loc='best')
    plt.tight_layout()
    plt.xlabel('Time (samples)')
    plt.ylabel('Spectrum (Hz)')
    plt.grid(True)
    plt.savefig(filename+'.png', format='png', dpi=1200)

    plt.show()


    print('\n # Performance ',filename,' MAE = ', MAE)


    return MAE

def prediction_sliding_scan(model, data0, windows_size=6, windows_step_size=1):
    data=data0.values
    # print('data=',data[:windows_size+2])
    y_pred=[]
    y_true=[]
    start=0
    n_frames=data.shape[0]-windows_size

    # print(' n_frames=',n_frames)

    for i in range(n_frames):
        stop=start+windows_size;
        x=data[start:stop]
        x=x.T;
        yreal=data[stop][0];
        y_true.append(yreal)

        # predict
        xin=np.array([np.flipud(x[0])])

        y = model.predict(xin)
        y_pred.append(y[0])

        # compare
        print('xin=',xin);
        print( 'y=',y,'\n')
        print( 'y real=',yreal,'\n')

        # clide the windows
        start=start+windows_step_size

        # input('flag')

    #plot the cv_results

    y_pred=np.asarray(y_pred); print('y_pred=',y_pred[:4])
    y_true=np.asarray(y_true); print('y_true=',y_true[:4])

    plot_prediction(y_true, y_pred,filename='scaning')

print('\n###################################################')
print('\n##      Bandwidth occupancy project 2020         ##')
print('\n###################################################')

###############################################################################
#%% input parameters
filename='data/rng_0.txt'
Type_feature='[Band occupancy]'
test_size=0.5
deploy_size=0.4
windows_size=10
windows_step_size=1
####################################################################################
data= load_bin_file(filename,max_lines=1e3)
# sns.pairplot(data)
data_, data_min, data_max =normalize_shuffle(data,norm=1)
print(data_.head())

label_col = 'y'
dataset=sliding_windows_series(data_, windows_size=windows_size,label_col = label_col)
print(dataset.head())

# extract out the features and labels into separate variables
y = dataset[label_col].values
data = dataset.drop(label_col, axis=1)
X = data.values
feature_names = data.columns
print('X: ', X.shape, '\n', X[:10,:])
print('y: ', y.T[:10])

model = run_ML_models(X, y,data_min, data_max, deploy_size=deploy_size, test_size=test_size, plot_intervals=True)

## implementation and deployment using sliding frames
# data_real=data_[:windows_size+50];
# print('\n Signal to predict using=',data_real,'\n Framesize=', windows_size)
# prediction_sliding_scan(model, data_real, windows_size=windows_size, windows_step_size=windows_step_size)
# train_X, test_X, train_y, test_y = timeseries_train_test_split(X, y, test_size=0.4)

print('############   THE END    #############')
