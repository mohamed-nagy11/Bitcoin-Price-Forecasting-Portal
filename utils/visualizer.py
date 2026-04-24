import plotly.graph_objects as go
import streamlit as st

def plot_forecast(historical_df, forecast_df, model_name="Prophet", df_moving_avg=None, active_moving_avg=None):
    """
    Generates an interactive Plotly chart for historical data, 
    future forecast, confidence intervals, and optional moving average
    """
    fig = go.Figure()

    # Historical Data (Actual Prices)
    fig.add_trace(go.Scatter(
        x=historical_df['ds'],
        y=historical_df['y'],
        mode='lines',
        name='Historical Actual Price',
        line=dict(color='red', width=2)
    ))

    # Moving Averages (Optional)
    if df_moving_avg is not None and active_moving_avg is not None:
        if active_moving_avg.get('SMA_7'):
            fig.add_trace(go.Scatter(
                x=df_moving_avg['ds'], y=df_moving_avg['SMA_7'], 
                mode='lines', name='7-Day SMA', 
                line=dict(color='magenta', width=1.5)
                ))
        if active_moving_avg.get('SMA_30'):
            fig.add_trace(go.Scatter(
                x=df_moving_avg['ds'], y=df_moving_avg['SMA_30'], 
                mode='lines', name='30-Day SMA', 
                line=dict(color='turquoise', width=1.5)
                ))
        if active_moving_avg.get('EMA_7'):
            fig.add_trace(go.Scatter(
                x=df_moving_avg['ds'], y=df_moving_avg['EMA_7'], 
                mode='lines', name='7-Day EMA', 
                line=dict(color="rgba(218, 165, 32, 1)", width=1.5)
                ))
        if active_moving_avg.get('EMA_30'):
            fig.add_trace(go.Scatter(
                x=df_moving_avg['ds'], y=df_moving_avg['EMA_30'], 
                mode='lines', name='30-Day EMA', 
                line=dict(color="rgba(138, 43, 226, 1)", width=1.5)
                ))

    # Confidence Interval
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'].tolist() + forecast_df['ds'].tolist()[::-1],
        y=forecast_df['yhat_upper'].tolist() + forecast_df['yhat_lower'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(0, 176, 246, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name=f'{model_name} Confidence Interval'
    ))

    # Future Forecast
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'], 
        y=forecast_df['yhat'],
        mode='lines',
        name=f'{model_name} Predicted Price',
        line=dict(color='blue', width=2, dash='dash')
    ))

    # Vertical line separator
    fig.add_vline(
        x=historical_df['ds'].iloc[-1],
        line_width=1,
        line_dash="dash",
        line_color="rgba(255,255,255,0.8)",
    )

    fig.update_layout(
        title=f"Bitcoin Price Forecast ({model_name})",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        template="plotly_dark",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig