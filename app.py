import streamlit as st
from utils import data_handler, visualizer
from models import prophet_model, arima_model

st.title("BTC Price Forecast Portal")

# File Uploader
uploaded_file = st.file_uploader("Upload your Historical BTC CSV", type=['csv'])

if uploaded_file:
    # Data Processing
    raw_df = data_handler.load_csv(uploaded_file)

    date_col = data_handler.detect_date_column(raw_df)
    
    if date_col is None:
        st.warning("⚠️ Could not automatically detect the Date column")
        date_col = st.selectbox("Select the Date column manually", raw_df.columns)
    else:
        st.success(f"✅ Auto-detected Date column: **{date_col}**")
        
    available_price_cols = data_handler.get_price_columns(raw_df, date_col)
    price_col = st.selectbox("Select Price Value to Forecast", available_price_cols)
    
    st.divider()

    # Model Parameters
    st.subheader("Forecast Parameters")
    selected_model = st.selectbox("Select Forecasting Model", ["Prophet", "ARIMA"])
    horizon = st.slider("Forecast Horizon (Days)", min_value=7, max_value=90, value=30)
    confidence_interval = st.slider("Confidence Interval (%)", min_value=80, max_value=99, value=95)
    
    st.write("**Technical Indicators**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        show_sma_7 = st.checkbox("7-Day SMA")
    with col2:
        show_sma_30 = st.checkbox("30-Day SMA")
    with col3:
        show_ema_7 = st.checkbox("7-Day EMA")
    with col4:
        show_ema_30 = st.checkbox("30-Day EMA")

    active_moving_avg = {
        'SMA_7': show_sma_7,
        'SMA_30': show_sma_30,
        'EMA_7': show_ema_7,
        'EMA_30': show_ema_30
    }

    # Execution Pipeline
    if st.button(f"Run {selected_model} Engine"):
        with st.spinner(f"Cleaning data and training {selected_model}..."):
            
            try:
                # Cleaning
                clean_df = data_handler.process_data(raw_df, date_col, price_col)

                # Backtesting and Forecasting
                if selected_model == "Prophet":
                    mae, rmse = prophet_model.evaluate_prophet(clean_df, horizon)
                    forecast_df = prophet_model.forecast_prophet(clean_df, horizon, confidence_interval)
                elif selected_model == "ARIMA":
                    mae, rmse = arima_model.evaluate_arima(clean_df, horizon)
                    forecast_df = arima_model.forecast_arima(clean_df, horizon, confidence_interval)

                # Moving Averages
                df_moving_avg = None
                if any(active_moving_avg.values()):
                    df_moving_avg = data_handler.calculate_moving_averages(clean_df)

                st.success(f"✅ {selected_model} Engine Execution Complete")
                
                # === Test Outputs ===
                
                st.subheader("Backtesting Metrics")
                st.write(f"**Mean Absolute Error (MAE):** ${mae:,.2f}")
                st.write(f"**Root Mean Squared Error (RMSE):** ${rmse:,.2f}")
                
                st.subheader("Interactive Forecast Chart")
                
                # Render the charts
                fig = visualizer.plot_forecast(
                    clean_df, 
                    forecast_df, 
                    model_name=selected_model, 
                    df_moving_avg=df_moving_avg, 
                    active_moving_avg=active_moving_avg
                    )
                st.plotly_chart(fig, use_container_width=True)
                
                # Toggle to see the raw numbers
                with st.expander("View Raw Forecast Data"):
                    st.dataframe(forecast_df.tail(horizon))
                
            except ValueError as e:
                # Catching the Unix timestamp / wrong column error
                st.error(f"❌ {e}")
                st.info("Please adjust your column selections and try again")
                st.stop()