# Bitcoin Price Forecasting Portal

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

An interactive, end-to-end Machine Learning web application designed to forecast Bitcoin prices. This portal empowers users to upload historical BTC data, dynamically configure prediction parameters, and generate robust time-series forecasts with seamless visualizations.

## Project Overview

The Bitcoin Price Forecasting Portal is a comprehensive, production-ready Machine Learning pipeline tailored for time-series forecasting. It integrates advanced forecasting algorithms, specifically highlighting the use of **Facebook Prophet** for capturing complex seasonal trends and **ARIMA** (AutoRegressive Integrated Moving Average) for dynamic statistical modeling. Built with a focus on data science best practices, it provides an intuitive interface for both exploratory data analysis and reliable future predictions.

## Key Features

- **Automated Data Cleaning:** Seamlessly detects date columns, handles Unix timestamps, and missing days using forward-fill.
- **Dynamic Backtesting:** Evaluates model performance using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
- **Interactive Visualizations:** Renders responsive, exploratory charts powered by Plotly, featuring confidence intervals and clear distinction between historical and predicted values.
- **Technical Indicators:** Supports dynamic overlay of Simple Moving Averages (SMA) and Exponential Moving Averages (EMA) directly onto the price charts.
- **CSV Export:** Enables users to easily download the generated forecast data for external reporting and analysis.
- **Responsive UI:** A clean, intuitive Streamlit-powered dashboard that adapts dynamically based on user input and error states.

## Project Architecture

The application is structured to enforce strong separation of concerns, maintaining decoupled modules for data processing, modeling, and visualization.

```text
Bitcoin-Price-Forecasting-Portal
┣ app.py                 # Main Streamlit application orchestrating the UI and user inputs
┣ models/                # Machine Learning implementations
┃ ┣ arima_model.py       # Auto-ARIMA for automatic parameter discovery and forecasting
┃ ┗ prophet_model.py     # Facebook Prophet configuration for capturing seasonal shifts
┗ utils/                 # Utility scripts and helper functions
  ┣ data_handler.py      # Resampling, auto-detection, and moving average calculations
  ┗ visualizer.py        # Generation of complex Plotly graphical objects  
```

## Installation & Setup

To run this application locally, ensure you have Python installed, then follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mohamed-nagy11/Bitcoin-Price-Forecasting-Portal.git
   cd Bitcoin-Price-Forecasting-Portal
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # On macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate

   # On Windows
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the application:**
   ```bash
   streamlit run app.py
   ```

## Usage Guide

Running a forecast is straightforward and requires only three easy steps:

1. **Upload Data:** Use the sidebar to upload your historical Bitcoin dataset (CSV format). 
2. **Configure Parameters:** Review the automatically detected date and price columns. Select your preferred model (Prophet or ARIMA), adjust the **Forecast Horizon** (7 to 90 days), and enable any desired Moving Averages.
3. **Generate Forecast:** Click **"Run Engine"**. The portal will automatically clean the data, backtest the model, display key metrics (MAE/RMSE), and plot an interactive forecast that you can optionally download as a CSV file.
