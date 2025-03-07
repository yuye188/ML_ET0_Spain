# Transfer and Deep learning models for daily reference evapotranspiration estimation and forecasting in Spain from local to national scale  **(Data and Code)**  

## Yu Ye,Aurora González-Vidal,Miguel A. Zamora-Izquierdo,Antonio F. Skarmeta
#### *Department of Information and Communication Engineering, University of Murcia, Murcia, 30100, Spain*

The structure of this repository is organized as follows:
* ```all data murcia```: Real meteorological and ET0 data in the Region of Murcia, obtained from http://siam.imida.es/.
* ```all data siar```: Real meteorological and ET0 data in Spain, obtained from https://servicio.mapa.gob.es/websiar/AltaUsuario.aspx?dst=3.
* ```forecastMurcia```: Forecast meterological data in the Region of Murcia, obtained from https://www.weatherbit.io/.
* ```forecastSiar```: Forecast meterological data in Spain, obtained from https://open-meteo.com/.
* ```createStandarMLmodels.ipynb```: The code to create the standard ML models (RF, SVR and MLP).
* ```createLSTMmodelsForTL.ipynb```: The code to create the TL models (LSTM).
* ```transferLearningCompareTLNoTL.py```: The code to get results from models that use TL or not.
* ```tlScriptCompareTLNoTL.bat```: Script to call the transferLearningCompareTLNoTL.py.
* ```plots.ipynb```: Notebook to create the figures.
* ```getSIARData.ipynb```: Notebook to download data from SIAR.
* ```<stationcode>M*.keras```: LSTM models trained from `<stationcode>`.
* ```finalModels<stationcode>NO-TL.pkl```: Standard models trained from `<stationcode>`.
* ```all_medidas<stationcode>.xlsx```: Complete local results of the station `<stationcode>`.
* ```all_scalers<stationcode>.pkl```: Scales for the data from `<stationcode>`.