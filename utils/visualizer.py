import plotly.graph_objects as go
import streamlit as st

def plot_forecast(historical_df, forecast_df, model_name="Prophet"):
    """
    Generates an interactive Plotly chart for historical data, 
    future forecast, and confidence intervals
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