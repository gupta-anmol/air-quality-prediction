## Air-quality-prediction
# Overview
- Prediction of pollution indicators - PM2.5, PM10, CO, SO2 and NO2
- Predicting in 1/2/3 days in the future using past 24 hours data only
- trained using keras with tensorflow as backend

Dependencies Required 
- scikit-learn
- keras tensorflow
- numpy, pandas, matplotlib, seaborn and tqdm

### Steps for training model 
- open script.ipynb and follow instructions 
- change hyperparamters in trainmodel functions

### Steps for using pretrained Model
- functions provided in demo.py for preprocessing and predicting usnig trained model and printing metrics and ploting graphs
- `run_CNN(pollutant='PM2.5',days=1)` 
- `run_ANN(pollutant='PM2.5',days=1)` 
- `run_SVM(pollutant='PM2.5',days=1)` 
- `run_LR(pollutant='PM2.5',days=1)` 
- pollutant strings parameters from 'PM2.5', 'PM10', 'CO', 'NO2', 'SO2'
- days can be 1,2 or 3

