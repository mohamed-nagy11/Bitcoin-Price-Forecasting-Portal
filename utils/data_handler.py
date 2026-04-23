import pandas as pd
import streamlit as st

@st.cache_data
def load_csv(uploaded_file):
    """
    Reads the uploaded CSV file into a Pandas DataFrame
    Chaches the data to run only once per unique uploaded_file
    """
    return pd.read_csv(uploaded_file)

def detect_date_column(df):
    """Automatically detects the timestamp column"""
    common_date_names = ['date', 'timestamp', 'datetime', 'time']
    
    for col in df.columns:
        if col.lower() in common_date_names:
            return col
            
    raise ValueError("Could not automatically detect a Date or Timestamp column. \
                    Please ensure your CSV has a column named 'Date' or 'Timestamp'.")

def detect_date_column(df):
    """
    Attempts to automatically identify the timestamp column
    Returns the column name if found, otherwise returns None
    """
    common_date_names = ['date', 'timestamp', 'datetime', 'time']
    for col in df.columns:
        if col.lower() in common_date_names:
            return col
            
    return None

def get_price_columns(df, date_col):
    """Returns a list of numeric columns, excluding the date column"""
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if date_col in numeric_cols:
        numeric_cols.remove(date_col)
        
    return numeric_cols

def format_prophet_columns(df, date_col, price_col):
    """Isolates the required columns and renames them to Prophet format ('ds' and 'y')"""
    df_subset = df[[date_col, price_col]].copy()
    df_subset.columns = ['ds', 'y']
    return df_subset

def parse_and_sort_dates(df):
    """Parses dates, handles corrupted rows, and sorts chronologically"""
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    df = df.dropna(subset=['ds'])
    return df.sort_values(by='ds')

def fill_missing_days(df):
    """Removes duplicate timestamps, resamples to daily frequency, and  applies forward-fill"""
    
    df = df.drop_duplicates(subset=['ds'], keep='last')
    df.set_index('ds', inplace=True)
    df = df.resample('D').ffill()
    df.reset_index(inplace=True)
    return df

@st.cache_data
def process_data(df, date_col, price_col):
    """
    Orchestrator function: Runs the data through the cleaning pipeline
    Decorated with @st.cache_data to run only once per unique uploaded_file
    """
    df_clean = format_prophet_columns(df, date_col, price_col)
    df_clean = parse_and_sort_dates(df_clean)
    df_clean = fill_missing_days(df_clean)
    return df_clean

def calculate_moving_averages(df):
    """Calculates 7-day and 30-day Simple Moving Averages"""
    df_tech = df.copy()
    df_tech['SMA_7'] = df_tech['y'].rolling(window=7).mean()
    df_tech['SMA_30'] = df_tech['y'].rolling(window=30).mean()
    return df_tech