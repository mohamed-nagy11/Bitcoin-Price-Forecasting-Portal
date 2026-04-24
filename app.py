import streamlit as st
from utils import data_handler, visualizer
from models import prophet_model, arima_model

st.set_page_config(page_title="BTC Forecast Portal", layout="wide")
st.title("BTC Price Forecast Portal")

# ===== Sidebaar =====

with st.sidebar:
    st.header("Configuration")
    
    # File Uploader
    uploaded_file = st.file_uploader("Upload your Historical BTC CSV", type=['csv'])

    if uploaded_file:
        st.divider()
        st.subheader("1. Data Selection")
        
        # Load data and detect columns
        raw_df = data_handler.load_csv(uploaded_file)
        auto_detected_col = data_handler.detect_date_column(raw_df)
        
        if auto_detected_col in raw_df.columns:
            default_index = list(raw_df.columns).index(auto_detected_col)
            st.success(f"✅ Auto-detected date: **{auto_detected_col}**")
        else:
            default_index = 0
            st.warning("⚠️ Could not auto-detect Date")
        
        date_col = st.selectbox("Confirm Date Column", raw_df.columns, index=default_index)
        
        available_price_cols = data_handler.get_price_columns(raw_df, date_col)
        price_col = st.selectbox("Select Price Value", available_price_cols)
        
        st.divider()
        st.subheader("2. Forecast Parameters")
        selected_model = st.selectbox("Select Model", ["Prophet", "ARIMA"])
        horizon = st.slider("Forecast Horizon (Days)", min_value=7, max_value=90, value=30)
        confidence_interval = st.slider("Confidence Interval (%)", min_value=80, max_value=99, value=95)
        
        with st.expander("Moving Averages"):
            col1, col2 = st.columns(2)
            with col1:
                show_sma_7 = st.checkbox("7-Day SMA")
                show_ema_7 = st.checkbox("7-Day EMA")
            with col2:
                show_sma_30 = st.checkbox("30-Day SMA")
                show_ema_30 = st.checkbox("30-Day EMA")

        active_moving_avg = {
            'SMA_7': show_sma_7, 'SMA_30': show_sma_30,
            'EMA_7': show_ema_7, 'EMA_30': show_ema_30
        }

        st.divider()
        run_button = st.button(f"Run {selected_model} Engine", use_container_width=True)


# ===== Main Area =====

if not uploaded_file:
    st.info("Please upload your historical BTC CSV file in the sidebar to get started")

elif uploaded_file and run_button:
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
            
            # === Test Outputs ===
            
            st.subheader("Backtesting Metrics")
            
            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric("Mean Absolute Error (MAE)", f"${mae:,.2f}")
            metric_col2.metric("Root Mean Squared Error (RMSE)", f"${rmse:,.2f}")
            
            st.subheader("Interactive Forecast Chart")
            
            # === Render Charts ===
            fig = visualizer.plot_forecast(
                clean_df, 
                forecast_df, 
                model_name=selected_model, 
                df_moving_avg=df_moving_avg, 
                active_moving_avg=active_moving_avg
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # ===== Download Forecast =====
            csv_data = forecast_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Forecast Data as CSV",
                data=csv_data,
                file_name=f"{selected_model}_btc_forecast.csv",
                mime="text/csv"
            )
            
            # ===== Raw Data =====
            with st.expander("View Raw Forecast Data"):
                st.dataframe(forecast_df.tail(horizon))
            
        except ValueError as e:
            st.error(f"❌ {e}")
            st.info("Please adjust your column selections and try again")