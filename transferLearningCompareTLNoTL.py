import pandas as pd
import numpy as np
import math
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
import random
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import keras
from tensorflow.keras.layers import LSTM, Dense

from keras.models import Sequential 
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping 
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import RandomizedSearchCV
from keras.models import load_model
import pickle
import datetime
import sys
import os

random.seed(123)
np.random.seed(123)
tf.random.set_seed(123)

print(datetime.datetime.now(), 'started.')



# Select station and TLPercentage to do the transfer learning
stationCode = sys.argv[1]
TLPercentage = float(sys.argv[2])

# path of murcia and siar data
stationPath = './all data murcia/' + stationCode + '.csv'
locSiar = pd.read_excel('./locations.xlsx', sheet_name='Hoja2', usecols='B:D', ).dropna()

# Define the best combination of each number of variables
param4 = ['TMAX', 'TMIN', 'HRMAX', 'HRMIN', 'RADMED', 'VVMED']
param3 = ['TMAX', 'TMIN', 'RADMED', 'VVMED']
param2 = ['TMAX', 'TMIN', 'RADMED']
param1 = ['RADMED']

all_params = [param4, param3, param2, param1]

# months of each season
mesesEstaciones =  [[4,5,6], # Spring
                    [7,8,9], # Summer
                    [10,11,12], # Autumn
                    [1,2,3]] # Winter

seasons = ['Spring', 'Summer', 'Autumn', 'Winter']

# Load standar ML models of selected station
with open('finalModels'+stationCode+'NO-TL.pkl', 'rb') as file:
    no_TL_models = pickle.load(file)

# Load scalers
with open('./all_scalers'+stationCode+'.pkl', 'rb') as file:
    all_scalers = pickle.load(file)

# Functions to read the station data
def convertirComa(x):
    if type(x) == str:
        return x.replace(",", ".")
    else:
        return x
def leerEstacionDatos(path):
    estacionDatas = pd.read_csv(path, encoding='ISO-8859-1', sep=";")
    estacionDatas.columns = ['ESTACION', 'MUNICIPIO', 'PARAJE', 'HORAS', 'FECHA', 'ETO','TMAX', 'TMIN', 'HRMAX', 'HRMIN', 'RADMED','VVMED', '-']
    estacionDatas = estacionDatas.drop(columns=['ESTACION', 'MUNICIPIO', 'PARAJE', 'HORAS', '-'])
    estacionDatas = estacionDatas.reset_index().drop(columns='index')
    estacionDatas['FECHA'] = pd.to_datetime(estacionDatas['FECHA'], format="%d/%m/%y")
    estacionDatas.index = estacionDatas['FECHA']
    estacionDatas.drop(columns='FECHA', inplace=True)
    estacionDatas.dropna(inplace=True)
    for i in estacionDatas.columns:
        estacionDatas[i] = pd.to_numeric(estacionDatas[i].apply(lambda x : convertirComa(x)))
    return estacionDatas
estacionDatas = leerEstacionDatos(stationPath)
estacionDatas

# SIAR stations
dirSiar = './all data siar/'
ficheros = os.listdir(dirSiar)
estacionesSiar = []
nombreEstacionesSiar = []
for f in ficheros:
    if 'csv' not in f:
        continue
    df = pd.read_csv(dirSiar+f)
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df.set_index('Fecha',inplace=True)
    df.columns = ['TMAX', 'TMIN', 'HRMAX', 'HRMIN', 'VVMED', 'RADMED', 'ETO']
    df.dropna(inplace=True)
    estacionesSiar.append(df)
    nombreEstacionesSiar.append(f.split('.')[0])

# Stations from Murcia Region
murciaDir = './all data murcia/'
nombreEstacionesMurcia = []
estacionesMurcia = []
for f in os.listdir(murciaDir):
    if stationCode in f:
        continue
    df = leerEstacionDatos(murciaDir+f)
    df = df[df.index >= '2017-01-01'] #2017
    df = df[df.index <= '2023-06-17']
    estacionesMurcia.append(df)
    nombreEstacionesMurcia.append(f.split('.')[0])


# Function that read the forecast data (hourly) of a specific station and convert it into daily values according each variable:
# T, Hr -> max and min
# U2 and Rs -> mean
def leerPredicciones(path):
    df = pd.read_csv(path)
    df['dates'] = pd.to_datetime(df['dates'])
    df.drop(columns=['Estacion', 'Servicio'], inplace=True)
    # Sacar el DF de WB agrupado por dias y con las variables calculadas 
    punto = [l[1] for l in list(df.groupby([df['dates'].dt.date]))]
    FECHA = []
    TMAX = []
    TMIN = []
    HRMAX = []
    HRMIN = []
    VVMED = []
    RADMED = []
    for p in punto:
        FECHA.append(pd.to_datetime(p['dates']).dt.date.iloc[0])
        TMAX.append(p['temp'].max())
        TMIN.append(p['temp'].min())
        HRMAX.append(p['rh'].max())
        HRMIN.append(p['rh'].min())
        VVMED.append(p['wind'].mean())
        RADMED.append(p['solar_rad'].mean())

    return pd.DataFrame({
        "FECHA": pd.to_datetime(FECHA),
        "TMAX": TMAX,
        "TMIN": TMIN,
        "HRMAX": HRMAX,
        "HRMIN": HRMIN,
        "VVMED": VVMED,
        "RADMED": RADMED
    }
    )

# Function to read forecasted climatic data in Murcia (WeatherBit or WB)
def leerPredictionTest():
    dir = './forecastMurcia/'
    subdirs = os.listdir(dir) # Name of stations

    dfs_wb = []
    for subdir in subdirs:

        # Fichero ETo real
        station = subdir.split('-')[0]
        eto = leerEstacionDatos('./all data murcia/' + station + '.csv')
        eto = eto[eto.index >= '2023-06-18']
        eto.reset_index(inplace=True)

        loc = dir+subdir+'/'

        # Ficheros de WB 
        df_wb = leerPredicciones(loc + 'WB-'+ subdir + '.csv')
        df_wb = pd.merge(df_wb, eto[['FECHA', 'ETO']], on='FECHA')
        df_wb['VVMED'] = df_wb['VVMED'].apply(lambda x: x*4.87/np.log(67.8*10-5.42))
        df_wb = df_wb[df_wb['FECHA'] <= '2024-06-25']
        df_wb.index = df_wb['FECHA']
        df_wb.drop(columns='FECHA', inplace=True)

        dfs_wb.append(df_wb)

    return dfs_wb, subdirs

dfs_wb, locs = leerPredictionTest()

# Function which converts hourly data to daily data
def hourlyToDaily(df):
    punto = [l[1] for l in list(df.groupby([df['dates'].dt.date]))]
    FECHA = []
    TMAX = []
    TMIN = []
    HRMAX = []
    HRMIN = []
    VVMED = []
    RADMED = []
    for p in punto:
        FECHA.append(pd.to_datetime(p['dates']).dt.date.iloc[0])
        TMAX.append(p['temp'].max())
        TMIN.append(p['temp'].min())
        HRMAX.append(p['rh'].max())
        HRMIN.append(p['rh'].min())
        VVMED.append(p['wind'].mean())
        RADMED.append(p['solar_rad'].mean())

    return pd.DataFrame({
        "FECHA": pd.to_datetime(FECHA),
        "TMAX": TMAX,
        "TMIN": TMIN,
        "HRMAX": HRMAX,
        "HRMIN": HRMIN,
        "VVMED": VVMED,
        "RADMED": RADMED
    })

# read the real and forecasted data for SIAR stations
def getLastYearSiarData(locSiar):
    siarLastYear = []
    for index, row in locSiar.iterrows():
        stationCode = row['Name']
        

        siarPred = pd.read_csv('./forecastSiar/'+stationCode+'.csv')
        siarPred.columns = ['dates', 'temp', 'rh', 'wind', 'solar_rad']
        siarPred['dates'] = pd.to_datetime(siarPred['dates'])
        siarPred['wind'] = siarPred['wind'].apply(lambda x: x*4.87/np.log(67.8*10-5.42))
        siarPred = hourlyToDaily(siarPred)
        
        siarReal = pd.read_csv('./siarRealDataForForecast/'+stationCode+'.csv')
        siarReal.columns = ['FECHA', 'TMAX', 'TMIN', 'HRMAX', 'HRMIN', 'VVMED', 'RADMED', 'ETO']
        siarReal['FECHA'] = pd.to_datetime(siarReal['FECHA'])
        siarReal.dropna(inplace=True)
        siarReal = siarReal[['FECHA', 'ETO']]

        df_join = pd.merge(siarReal, siarPred, how='inner', on='FECHA')
        df_join.set_index('FECHA',inplace=True)
        siarLastYear.append(df_join)

    return siarLastYear

siarLastYear = getLastYearSiarData(locSiar)

def getMonths(input, meses):
    return input.loc[(input.index.month==meses[0]) | (input.index.month==meses[1]) | (input.index.month==meses[2])]


# Function to apply transfer learning
def transferLearning4estaciones(dfs, names, stationName, percentageTestSize, comb, scalers, tfmodelName):

    # dataframes to store results of NoTL, TL with train layers and TL without train layers
    all_seasons_notl = pd.DataFrame()
    all_seasons_tl_trainLayers = pd.DataFrame()
    all_seasons_tl_NotrainLayers = pd.DataFrame()
    
    for idx, station in enumerate(dfs):

        season = []


        # array to store metrics of each of three method
        R2_notl = []
        MAE_notl = []
        MAPE_notl = []
        RMSE_notl = []

        R2_tl_trainLayers = []
        MAE_tl_trainLayers = []
        MAPE_tl_trainLayers = []
        RMSE_tl_trainLayers = []

        R2_tl_NotrainLayers = []
        MAE_tl_NotrainLayers = []
        MAPE_tl_NotrainLayers = []
        RMSE_tl_NotrainLayers = []
        
        # iterate each season
        for i, meses in enumerate(mesesEstaciones):

            # get df of the corresponding season
            dfEstacion = getMonths(station, meses)
            
            # if the df has less than 30 days of data, continue to the next season
            if len(dfEstacion) < 30:
                continue
            
            season.append(seasons[i])

            # scale and split the data
            X_scaled = scalers[0].transform(dfEstacion[comb])
            y_scaled = scalers[1].transform(np.array(dfEstacion['ETO']).reshape(-1, 1))

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=percentageTestSize, shuffle=False)
            
            print('Comb:', comb, ', Station', names[idx], ', months:', meses ,', season:',seasons[i], ', train:', len(X_train), ', test:', len(X_test))
            y_test = scalers[1].inverse_transform(y_test).flatten()
            
            # No Transfer Learning Models (standard ML models)
            no_TL_model = no_TL_models[4-int(tfmodelName[1])]
            predictions = no_TL_model.predict(pd.DataFrame(X_test, columns=comb)).flatten() # make the prediction
            r2 = np.corrcoef(y_test, predictions)[0][1]**2
            mae = mean_absolute_error(y_true=y_test,y_pred=predictions)
            mape = mean_absolute_percentage_error(y_true=y_test,y_pred=predictions)*100
            rmse = mean_squared_error(y_true=y_test,y_pred=predictions,squared=False)

            # save the metrics
            R2_notl.append(r2)
            MAE_notl.append(mae)
            MAPE_notl.append(mape)
            RMSE_notl.append(rmse)
            
            print('Without TL', names[idx], meses, 'R2=',r2, 'MAE=',mae, 'MAPE=',mape)

            # Reshape sets for LSTM layers
            X_train = np.array(X_train).reshape(X_train.shape[0], 1, X_train.shape[1])
            X_test = np.array(X_test).reshape(X_test.shape[0], 1, X_test.shape[1])
            y_train = np.array(y_train).reshape(y_train.shape[0], 1)


            # Transfer Learning retraining layers
            tfModel = load_model('./' + stationName + tfmodelName+'.keras') # load the model
            tfModel.compile(optimizer=tfModel.optimizer, loss=tfModel.loss) # compile the model
            tfModel.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0, callbacks=[EarlyStopping(monitor = 'val_loss', patience = 15)]) # retrain the model
            
            predictions = tfModel.predict(X_test, verbose=0) # make predictions
            predictions = scalers[1].inverse_transform(predictions).flatten()
            
            
            r2 = np.corrcoef(y_test, predictions)[0][1]**2
            mae = mean_absolute_error(y_true=y_test,y_pred=predictions)
            mape = mean_absolute_percentage_error(y_true=y_test,y_pred=predictions)*100
            rmse = mean_squared_error(y_true=y_test,y_pred=predictions,squared=False)

            print('With TL-TrainLayers', names[idx], meses, 'R2=',r2, 'MAE=',mae, 'MAPE=',mape)

            # save the metrics
            R2_tl_trainLayers.append(r2)
            MAE_tl_trainLayers.append(mae)
            MAPE_tl_trainLayers.append(mape)
            RMSE_tl_trainLayers.append(rmse)

            
            # Transfer Learning but NOT retrain layers
            tfModel = load_model('./' + stationName + tfmodelName+'.keras') # load the model

            # freeze the layers
            tfModel.layers[0].trainable = False
            tfModel.layers[1].trainable = False
            tfModel.layers[2].trainable = False
            tfModel.compile(optimizer=tfModel.optimizer, loss=tfModel.loss) # compile the model
            tfModel.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0, callbacks=[EarlyStopping(monitor = 'val_loss', patience = 15)]) # retrain the model
            
            predictions = tfModel.predict(X_test, verbose=0) # make predictions
            predictions = scalers[1].inverse_transform(predictions).flatten()
            
            
            r2 = np.corrcoef(y_test, predictions)[0][1]**2
            mae = mean_absolute_error(y_true=y_test,y_pred=predictions)
            mape = mean_absolute_percentage_error(y_true=y_test,y_pred=predictions)*100
            rmse = mean_squared_error(y_true=y_test,y_pred=predictions,squared=False)

            print('With TL-noTrainLayers', names[idx], meses, 'R2=',r2, 'MAE=',mae, 'MAPE=',mape)
            print()

            # save the metrics
            R2_tl_NotrainLayers.append(r2)
            MAE_tl_NotrainLayers.append(mae)
            MAPE_tl_NotrainLayers.append(mape)
            RMSE_tl_NotrainLayers.append(rmse)

        # create the dataframes of each method
        df_station_notl=pd.DataFrame({'Season': season,
                                 'R2': R2_notl,
                                 'MAE': MAE_notl,
                                 'MAPE': MAPE_notl,
                                 'RMSE': RMSE_notl,
                                 'Station': names[idx]})
        all_seasons_notl = pd.concat([all_seasons_notl, df_station_notl])
            

        df_station_tl_trainLayers=pd.DataFrame({'Season': season,
                                 'R2': R2_tl_trainLayers,
                                 'MAE': MAE_tl_trainLayers,
                                 'MAPE': MAPE_tl_trainLayers,
                                 'RMSE': RMSE_tl_trainLayers,
                                 'Station': names[idx]})
        all_seasons_tl_trainLayers = pd.concat([all_seasons_tl_trainLayers, df_station_tl_trainLayers])


        df_station_tl_NotrainLayers=pd.DataFrame({'Season': season,
                                 'R2': R2_tl_NotrainLayers,
                                 'MAE': MAE_tl_NotrainLayers,
                                 'MAPE': MAPE_tl_NotrainLayers,
                                 'RMSE': RMSE_tl_NotrainLayers,
                                 'Station': names[idx]})
        all_seasons_tl_NotrainLayers = pd.concat([all_seasons_tl_NotrainLayers, df_station_tl_NotrainLayers])

    return all_seasons_notl, all_seasons_tl_trainLayers, all_seasons_tl_NotrainLayers

# main function that prepare the models/dataframes to call the transfer learning function
def getAllResults4estaciones(dfs, names, stationName, percentageTestSize, all_scalers, all_params):
    
    all_medidas_notl = pd.DataFrame()
    all_medidas_tl_trainLayers = pd.DataFrame()
    all_medidas_tl_NotrainLayers = pd.DataFrame()
    
    for idx,comb in enumerate(all_params):
        df_results_notl, df_results_tl_trainLayers, df_results_tl_NotrainLayers, = transferLearning4estaciones(dfs, names, stationName=stationName, percentageTestSize=percentageTestSize, comb=comb, 
                                                 scalers=all_scalers[idx], tfmodelName='M'+str(4-idx))
        
        df_results_notl['Model'] = 'M'+str(4-idx) 
        df_results_tl_trainLayers['Model'] = 'M'+str(4-idx) 
        df_results_tl_NotrainLayers['Model'] = 'M'+str(4-idx) 

        all_medidas_notl = pd.concat([all_medidas_notl, df_results_notl])
        all_medidas_tl_trainLayers = pd.concat([all_medidas_tl_trainLayers, df_results_tl_trainLayers])
        all_medidas_tl_NotrainLayers = pd.concat([all_medidas_tl_NotrainLayers, df_results_tl_NotrainLayers])
        
    return  all_medidas_notl, all_medidas_tl_trainLayers, all_medidas_tl_NotrainLayers




# Excute functions to get the test results of each scale (Murica/Spain)

# Estimation only for TLPercentage: 0.8 (0.2), 0.9 (0.1), 0.95 (0.05) and 0.99 (0.01)
if (TLPercentage >= 0.8) & (TLPercentage <= 0.99):
    # Murcia
    print(stationCode, TLPercentage, 'estimation Murcia started.')
    all_medidas_notl, all_medidas_tl_trainLayers, all_medidas_tl_NotrainLayers = getAllResults4estaciones(estacionesMurcia, nombreEstacionesMurcia, stationCode, TLPercentage, all_scalers, all_params)

    all_medidas_notl['TLPercentage'] = TLPercentage
    filepath = './'+stationCode+'/'+'MurciaEstimation_NoTL.csv'
    all_medidas_notl.to_csv(filepath, mode='a', header=True if os.path.exists(filepath) == False else False, index=False)

    all_medidas_tl_trainLayers['TLPercentage'] = TLPercentage
    filepath = './'+stationCode+'/'+'MurciaEstimation_TL-TrainLayers.csv'
    all_medidas_tl_trainLayers.to_csv(filepath, mode='a', header=True if os.path.exists(filepath) == False else False, index=False)
    
    all_medidas_tl_NotrainLayers['TLPercentage'] = TLPercentage
    filepath = './'+stationCode+'/'+'MurciaEstimation_TL-NoTrainLayers.csv'
    all_medidas_tl_NotrainLayers.to_csv(filepath, mode='a', header=True if os.path.exists(filepath) == False else False, index=False)
    
    print(stationCode, TLPercentage, 'estimation Murcia done.')
            
    # Esp
    print(stationCode, TLPercentage, 'estimation Spain started.')
    all_medidas_notl, all_medidas_tl_trainLayers, all_medidas_tl_NotrainLayers = getAllResults4estaciones(estacionesSiar, nombreEstacionesSiar, stationCode, TLPercentage, all_scalers, all_params)

    all_medidas_notl['TLPercentage'] = TLPercentage
    filepath = './'+stationCode+'/'+'EspEstimation_NoTL.csv'
    all_medidas_notl.to_csv(filepath, mode='a', header=True if os.path.exists(filepath) == False else False, index=False)

    all_medidas_tl_trainLayers['TLPercentage'] = TLPercentage
    filepath = './'+stationCode+'/'+'EspEstimation_TL-TrainLayers.csv'
    all_medidas_tl_trainLayers.to_csv(filepath, mode='a', header=True if os.path.exists(filepath) == False else False, index=False)
    
    all_medidas_tl_NotrainLayers['TLPercentage'] = TLPercentage
    filepath = './'+stationCode+'/'+'EspEstimation_TL-NoTrainLayers.csv'
    all_medidas_tl_NotrainLayers.to_csv(filepath, mode='a', header=True if os.path.exists(filepath) == False else False, index=False)
    
    print(stationCode, TLPercentage, 'estimation Spain done.')



# Forecast only for TLPercentage: 0.6-0.9 (0.4-0.1)
if (TLPercentage >= 0.6) & (TLPercentage <= 0.9):

    
    # Murcia
    print(stationCode, TLPercentage, 'Forecast Murcia started.')
    all_medidas_notl, all_medidas_tl_trainLayers, all_medidas_tl_NotrainLayers = getAllResults4estaciones(dfs_wb, locs, stationCode, TLPercentage, all_scalers, all_params)

    all_medidas_notl['TLPercentage'] = TLPercentage
    filepath = './'+stationCode+'/'+'MurciaForecast_NoTL.csv'
    all_medidas_notl.to_csv(filepath, mode='a', header=True if os.path.exists(filepath) == False else False, index=False)

    all_medidas_tl_trainLayers['TLPercentage'] = TLPercentage
    filepath = './'+stationCode+'/'+'MurciaForecast_TL-TrainLayers.csv'
    all_medidas_tl_trainLayers.to_csv(filepath, mode='a', header=True if os.path.exists(filepath) == False else False, index=False)
    
    all_medidas_tl_NotrainLayers['TLPercentage'] = TLPercentage
    filepath = './'+stationCode+'/'+'MurciaForecast_TL-NoTrainLayers.csv'
    all_medidas_tl_NotrainLayers.to_csv(filepath, mode='a', header=True if os.path.exists(filepath) == False else False, index=False)
    
    print(stationCode, TLPercentage, 'Forecast Murcia done.')
    

    # Esp
    print(stationCode, TLPercentage, 'Forecast Spain started.')
    all_medidas_notl, all_medidas_tl_trainLayers, all_medidas_tl_NotrainLayers = getAllResults4estaciones(siarLastYear, locSiar['Name'], stationCode, TLPercentage, all_scalers, all_params)

    all_medidas_notl['TLPercentage'] = TLPercentage
    filepath = './'+stationCode+'/'+'EspForecast_NoTL.csv'
    all_medidas_notl.to_csv(filepath, mode='a', header=True if os.path.exists(filepath) == False else False, index=False)

    all_medidas_tl_trainLayers['TLPercentage'] = TLPercentage
    filepath = './'+stationCode+'/'+'EspForecast_TL-TrainLayers.csv'
    all_medidas_tl_trainLayers.to_csv(filepath, mode='a', header=True if os.path.exists(filepath) == False else False, index=False)
    
    all_medidas_tl_NotrainLayers['TLPercentage'] = TLPercentage
    filepath = './'+stationCode+'/'+'EspForecast_TL-NoTrainLayers.csv'
    all_medidas_tl_NotrainLayers.to_csv(filepath, mode='a', header=True if os.path.exists(filepath) == False else False, index=False)
    
    print(stationCode, TLPercentage, 'Forecast Spain done.')

print(datetime.datetime.now(), 'finished.')