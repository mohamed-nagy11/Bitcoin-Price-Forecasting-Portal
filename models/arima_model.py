import pandas as pd
import numpy as np
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st

@st.cache_data
def evaluate_arima(df, horizon):
    """
    Splits the data into train/test sets dynamically based on the forecast horizon, 
    discovers the best ARIMA parameters automatically, and calculates MAE and RMSE metrics
    """
    train = df.iloc[:-horizon]
    test = df.iloc[-horizon:]
    
    model = pm.auto_arima(
        train['y'], 
        seasonal=False, 
        suppress_warnings=True, 
        stepwise=True
        )
    
    forecast = model.predict(n_periods=horizon)
    
    mae = mean_absolute_error(test['y'], forecast)
    rmse = np.sqrt(mean_squared_error(test['y'], forecast))
    
    return mae, rmse

@st.cache_data
def forecast_arima(df, horizon, confidence_interval):
    """
    Trains an Auto-ARIMA model on the FULL dataset 
    and generates future predictions with dynamic confidence intervals
    """    
    model = pm.auto_arima(
        df['y'], 
        seasonal=False, 
        suppress_warnings=True, 
        stepwise=True
        )
    
    forecast, conf_int = model.predict(
        n_periods=horizon, 
        return_conf_int=True, 
        alpha=1.0 - (confidence_interval / 100.0)
        )
    
    last_date = df['ds'].iloc[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), 
        periods=horizon, freq='D'
        )
    
    # Match the output format of Prophet
    forecast_df = pd.DataFrame({
        'ds': future_dates,
        'yhat': forecast.values,
        'yhat_lower': conf_int[:, 0],
        'yhat_upper': conf_int[:, 1]
    })
    
    return forecast_df