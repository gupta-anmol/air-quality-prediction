#!/usr/bin/env python
# coding: utf-8

# In[22]:


import air_prediction as air
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np
import keras
def predictStats(model,Xtrain,ytrain,Xtest,ytest):
    testPred = model.predict(Xtest)
    trainPred = model.predict(Xtrain)
    print("Test RMSE",mean_squared_error(testPred, ytest,squared=False))
    print("Train RMSE",mean_squared_error(trainPred, ytrain,squared=False))
    print("Test MAE",mean_absolute_error(testPred, ytest))
    print("Train MAE",mean_absolute_error(trainPred, ytrain))
def run_CNN(pollutant='PM2.5',days=1):
    met,aqi,split_aqi = air.loadcsv()
    dat = air.dataset(met,aqi,split_aqi)
    df = dat.mergedData('DL031',roll=48,shift=days*24)
    Xtrain,ytrain,Xtest,ytest = air.getSplitFeaturesTimeSeries(df,TIME_SERIES_LENGTH = 24,sel=pollutant)
    reconstructed_model = keras.models.load_model('./models/'+str(days)+"day"+pollutant+"_366")
    testPred = reconstructed_model.predict(Xtest)
    trainPred = reconstructed_model.predict(Xtrain)
    predictStats(reconstructed_model,Xtrain,ytrain,Xtest,ytest)
    air.plotPredvsTrue(testPred.flatten()[0:240],np.array(ytest).flatten()[0:240],'Actual vs Predicted','Unit of selected unit')
    # PLOT results of a random day
    x = np.random.randint(testPred.shape[0])
    air.plotPredvsTrue(testPred[x,:],np.array(ytest)[x,:],'Pollutant','unit')
#     air.getPearsonCorr(testPred.flatten(),np.array(ytest).flatten())
    
def run_ANN(pollutant='PM2.5',days=1):
    met,aqi,split_aqi = air.loadcsv()
    dat = air.dataset(met,aqi,split_aqi)
    df = dat.mergedData('DL031',roll=48,shift=48)
    Xtrain,ytrain,Xtest,ytest = air.getSplitFeatures(df,sel='SO2')
    # print(Xtrain.shape,ytrain.shape,Xtest.shape,ytest.shape)
    reconstructed_model = keras.models.load_model('./models/'+'ANN_'+str(days)+"day"+pollutant+"_366")
    testPred = reconstructed_model.predict(Xtest)
    trainPred = reconstructed_model.predict(Xtrain)
    predictStats(reconstructed_model,Xtrain,ytrain,Xtest,ytest)
#     getPearsonCorr(testPred,ytest)
    
def run_SVM(pollutant='PM2.5',days=1):
    met,aqi,split_aqi = air.loadcsv()
    dat = air.dataset(met,aqi,split_aqi)
    df = dat.mergedData('DL031',roll=48,shift=48)
    Xtrain,ytrain,Xtest,ytest = air.getSplitFeatures(df,sel='SO2')
    # print(Xtrain.shape,ytrain.shape,Xtest.shape,ytest.shape)
    reconstructed_model = pickle.loads(open('./models/'+'SVM_'+str(days)+"day"+pollutant+"_366",'rb').read())
    testPred = reconstructed_model.predict(Xtest)
    trainPred = reconstructed_model.predict(Xtrain)
    predictStats(reconstructed_model,Xtrain,ytrain,Xtest,ytest)
#     getPearsonCorr(testPred,ytest)
    
def run_LR(pollutant='PM2.5',days=1):
    met,aqi,split_aqi = air.loadcsv()
    dat = air.dataset(met,aqi,split_aqi)
    df = dat.mergedData('DL031',roll=48,shift=48)
    Xtrain,ytrain,Xtest,ytest = air.getSplitFeatures(df,sel='SO2')
    # print(Xtrain.shape,ytrain.shape,Xtest.shape,ytest.shape)
    reconstructed_model = pickle.loads(open('./models/'+'LR_'+str(days)+"day"+pollutant+"_366",'rb').read())
    testPred = reconstructed_model.predict(Xtest)
    trainPred = reconstructed_model.predict(Xtrain)
    predictStats(reconstructed_model,Xtrain,ytrain,Xtest,ytest)
#     getPearsonCorr(testPred,ytest)
    
run_CNN(pollutant='PM2.5',days=2)

