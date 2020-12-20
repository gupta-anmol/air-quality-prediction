RKP = "DL031"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
    
# Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import Dropout
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import Input
from keras.layers import RepeatVector
from keras.layers import BatchNormalization
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
import tensorflow as tf
import keras

#scipy
from scipy import stats

#tqdm
from tqdm import tqdm
sns.set_theme(style="darkgrid")


# Load datasets and rename columns, load all aqi data but specify metro data name
def loadcsv(city="./data/rkpuram.csv"):
    met = pd.read_csv(city,delimiter=';',skiprows=24)
    aqi = pd.read_csv('./data/station_hour.csv')
    print(aqi.columns)
    met.rename(columns={'# Date': 'Date',}, inplace=True)
    met.rename(columns={'UT time': 'Time',}, inplace=True)
    aqi['Time'] = aqi['Datetime'].str[-8:-3]
    aqi['Date'] = aqi['Datetime'].str[0:10]
    stations = ["DL"+str(x).zfill(3) for x in range(1,39)]
    split_aqi = {}
    for i in range(len(stations)):
        split_aqi[stations[i]] = (aqi[aqi['StationId'] == stations[i]])
    return met,aqi,split_aqi
met,aqi,split_aqi = loadcsv()


# # Merging and processing data

# Pre - processing and loading data
class dataset:
    def __init__(self,met,aqi,split_aqi):
            self.metro_data = met
            self.aqi_data = aqi
            self.split_aqi = split_aqi
    def mergedData(self,station,rlist=['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','AQI'],roll=48,shift=72):
        df_aqi = self.getdf(station)
        df = pd.merge(df_aqi, self.metro_data, how='inner', on=['Date', 'Time'])
        print("Merged Dataset Size",len(df))
        
        #Pre Processing merged Data
        df['Year'] = df['Date'].str[0:4]
        df['Month'] = df['Date'].str[5:7].astype(np.float64)
        df['Day'] = df['Date'].str[8:10].astype(np.float64)
        df['Hour'] = df['Time'].str[0:2]
        
        # TRIG TRANSFORMATIONS
        df['windX'] = np.cos(np.deg2rad(df['Wind direction'])) * df['Wind speed']
        df['windY'] = np.sin(np.deg2rad(df['Wind direction'])) * df['Wind speed']
        df['hourX'] = np.cos((df['Hour'].astype(np.float64)-1)*np.pi/24)
        df['hourY'] = np.sin((df['Hour'].astype(np.float64)-1)*np.pi/24)
        df['MonthX'] = np.cos((df['Month'].astype(np.float64)-1)*np.pi/12)
        df['MonthY'] = np.sin((df['Month'].astype(np.float64)-1)*np.pi/12)
        
        import datetime
        df['Date'] = pd.to_datetime(df['Date'])
        df['isWeekend'] =  (df['Date'].dt.dayofweek>=5).astype(int)
        
        df.interpolate(method='linear', limit=5,inplace=True)
        
        # Drop Additional columns
        df.drop('Benzene', axis=1, inplace=True)
        df.drop('Toluene',axis=1, inplace=True)
        df.drop('Xylene', axis=1,inplace=True)
        df.drop('AQI_Bucket',axis=1,inplace=True)
        df.drop('Datetime',axis=1,inplace=True)
        df.drop('StationId',axis=1,inplace=True)
        df.drop('Short-wave irradiation',axis=1,inplace=True)
        df.drop('Date',axis=1,inplace=True)
        df.drop('Time',axis=1,inplace=True)
        
        # Rolling and shifting 
        print("Size before roll",len(df))
        rollList = ['PM2.5','PM10','NO','NO2','CO','AQI','Temperature','Relative Humidity','windX','windY','Year','MonthX','MonthY','hourX','hourY','isWeekend']
        for i in rollList:
            df[i+'_lagroll1'] = df[i].rolling(window=24, min_periods=12).mean().shift(6)
            df[i+'_lagroll2'] = df[i].rolling(window=24, min_periods=12).mean().shift(12)
            df[i+'_lagroll3'] = df[i].rolling(window=24, min_periods=12).mean().shift(18)
            df[i+'_lagroll4'] = df[i].rolling(window=24, min_periods=12).mean().shift(24)
            
        for i in rlist:
            df[i+'_lag1'] = df[i].shift(24)
            df[i+'_lag2'] = df[i].shift(48)
            df[i+'_lag3'] = df[i].shift(72)
        for i in rlist:
            df[i+"_pred1"] = df[i].shift(-24)
            df[i+"_pred2"] = df[i].shift(-48)
            df[i+"_pred3"] = df[i].shift(-72)
        newlist = rlist + ['Temperature','Relative Humidity','windX','windY']
        for i in newlist:
            for j in range(24):
                df[i+"_t-"+str(j)] = df[i].shift(j)
                df[i+"_t+"+str(j)] = df[i].shift(-j-shift)
        futurelist = ['Year','MonthX','MonthY','hourX','hourY','isWeekend']
        for i in futurelist:
            for j in range(24):
                df[i+"_t-"+str(j)] = df[i].shift(-(shift+23-j))
        df.dropna(inplace=True)
        print("Size after roll",len(df))
        
        return df.copy()
    def getdf(self,station):
        return self.split_aqi[station]
    def plot(self,station):
        df = self.getdf(station)
    def stats(self):
        pass

# CNN Model Testing as well
def getSplitFeaturesTimeSeries(df,TIME_SERIES_LENGTH = 24,sel='PM2.5'):
    features = []
    rlist=['PM2.5','PM10','NO','NO2','CO','AQI']
#     for it in rlist:
#         print(it,np.mean(df[it]),np.std(df[it]))
    newlist = rlist + ['Temperature','Relative Humidity','windX','windY','Year','MonthX','MonthY','hourX','hourY','isWeekend']
    for j in range(24):
        for i in newlist:
            features.append(i+'_t-'+str(j))
    predVector = []
    for j in range(24):
        predVector.append(sel+'_t+'+str(j))
    X = df[features]
    y = df[predVector]
    X = np.array(X).reshape(X.shape[0],TIME_SERIES_LENGTH,len(newlist))
    scaler = StandardScaler()
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.33, random_state=42)
    Xtrain = scaler.fit_transform(Xtrain.reshape(Xtrain.shape[0],TIME_SERIES_LENGTH*len(newlist)))
    Xtrain = Xtrain.reshape(Xtrain.shape[0],TIME_SERIES_LENGTH,len(newlist))
    Xtest = scaler.transform(Xtest.reshape(Xtest.shape[0],TIME_SERIES_LENGTH*len(newlist)))
    Xtest = Xtest.reshape(Xtest.shape[0],TIME_SERIES_LENGTH,len(newlist))
    return Xtrain,ytrain,Xtest,ytest
def getSplitFeatures(df,sel='PM2.5'):
    features = []
    rlist=['PM2.5','PM10','NO','NO2','CO','AQI']
    for it in rlist:
        print(it,np.mean(df[it]),np.std(df[it]))
    newlist = rlist + ['Temperature','Relative Humidity','windX','windY']
    for j in range(1,5):
        for i in newlist:
            features.append(i+'_lagroll'+str(j))
    futurelist = ['Year','MonthX','MonthY','hourX','hourY','isWeekend']
    for j in futurelist:
        features.append(i+'_t-0')
    print("features length",len(newlist))
    predVector = [sel+'_t+0']
    X = df[features]
    y = df[predVector]
    X = np.array(X)
    scaler = StandardScaler()
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.33, random_state=42)
    Xtrain = scaler.fit_transform(Xtrain)
    Xtest = scaler.transform(Xtest)
    return Xtrain,ytrain,Xtest,ytest

def trainModel1(Xtrain,ytrain,Xtest,ytest):
    time_series = Xtrain.shape[1]
    features = Xtrain.shape[2]
    model = Sequential()
    model.add(Conv1D(128, 3,activation='relu',input_shape=(time_series,features)))
    model.add(BatchNormalization())
    model.add(Conv1D(128, 6,activation='relu',input_shape=(time_series-2,features)))
    model.add(BatchNormalization())
    model.add(Conv1D(128, 6,activation='relu',input_shape=(time_series-4,features)))
    model.add(BatchNormalization())

    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    
    model.add(Dense(24, activation='relu'))
    model.summary()
    #Fit
    model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
    history = model.fit(Xtrain, ytrain, epochs=200, batch_size=256,  verbose=1, validation_split=0.2)
    return model,history
def trainModel2(Xtrain,ytrain,Xtest,ytest):
    features = Xtrain.shape[1]
    model = Sequential()
    model.add(Input(shape=(features)))
    
    model.add(Dense(200, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(100, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(50, activation='relu'))
    model.add(BatchNormalization())
    
    model.add(Dense(1, activation='relu'))
    model.summary()
    #Fit
    model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
    history = model.fit(Xtrain, ytrain, epochs=200, batch_size=256,  verbose=1, validation_split=0.2)
    return model,history


# In[577]:


def predictStats(model,Xtrain,ytrain,Xtest,ytest):
    testPred = model.predict(Xtest)
    trainPred = model.predict(Xtrain)
    print("Test RMSE",mean_squared_error(testPred, ytest,squared=False))
    print("Train RMSE",mean_squared_error(trainPred, ytrain,squared=False))
    print("Test MAE",mean_absolute_error(testPred, ytest))
    print("Train MAE",mean_absolute_error(trainPred, ytrain))
def plothistory(history):
    # list all data in history
    print(history.history.keys())
    # summarize history for loss
    plt.plot(np.sqrt(history.history['loss']))
    plt.plot(np.sqrt(history.history['val_loss']))
    plt.title('model loss')
    plt.ylabel('RMSE loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.show()
def plotPredvsTrue(pred,true,title,ylabel):
    plt.figure(figsize=(15,7.5)) 
    plt.plot(true)
    plt.plot(pred)
    plt.ylabel(ylabel)
    plt.xlabel('Time Stamp Number')
    plt.legend(['Ground Truth', 'Prediction'], loc='upper right')
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.show()
def getPearsonCorr(pred,true):
    return stats.pearsonr(pred.flatten(),np.array(true).flatten())

