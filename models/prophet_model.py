import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st

@st.cache_data
def evaluate_prophet(df, horizon):
    """
    Splits the data into train/test sets dynamically based on the forecast horizon,
    trains the model, and calculates MAE and RMSE metrics
    """
    train = df.iloc[:-horizon]
    test = df.iloc[-horizon:]
    
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True
    )
    model.fit(train)
    
    forecast = model.predict(test[['ds']])
    
    mae = mean_absolute_error(test['y'], forecast['yhat'])
    rmse = np.sqrt(mean_squared_error(test['y'], forecast['yhat']))
    
    return mae, rmse

@st.cache_data
def forecast_prophet(df, horizon, confidence_interval):
    """
    Trains a Prophet model on the FULL dataset 
    and generates future predictions with dynamic confidence intervals
    """    
    model = Prophet(
        interval_width=confidence_interval / 100.0, 
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True
    )
    model.fit(df)
    
    future_dates = model.make_future_dataframe(periods=horizon, freq='D')
    
    forecast = model.predict(future_dates)
    
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]