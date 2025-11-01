import yfinance as yf
import pandas as pd
import numpy as np
from pmdarima import auto_arima
from datetime import date, timedelta
import warnings

# Suppress harmless warnings
warnings.filterwarnings("ignore")

def fetch_historical_data(ticker_symbol, start_date='2020-01-01'):
    """Fetches historical 'Close' price data for a given ticker."""
    try:
        # Get data up to the present day
        data = yf.download(ticker_symbol, start=start_date, end=date.today(), progress = False)
        
        if data.empty:
            print(f"Error: Could not retrieve data for ticker {ticker_symbol}. Check the ticker symbol.")
            return None

        # 1. SUCCESS LOGIC: Extract and clean the 'Close' data here (in the 'try' block)
        # Note: I'm assuming you want to return the full DataFrame for `predict_stock_price`
        # but if you only want the 'Close' series, adjust the return below.
        df = data[['Close']].copy()
        df.index = pd.to_datetime(df.index)
        return df # <--- Return the processed DataFrame on success
    
    except Exception as e:
        print(f"An error occurred during data fetch: {e}")
        return None

def predict_stock_price(df, n_periods=2):
    """
    Uses the Auto-ARIMA model to predict future stock prices.
    
    ARIMA (AutoRegressive Integrated Moving Average) is a powerful statistical 
    method for time series forecasting. Auto-ARIMA automatically selects the 
    best model parameters (p, d, q) based on the data.
    """
    print("-> Training Auto-ARIMA model...")
    # Auto-ARIMA to find the best model parameters
    model = auto_arima(
        df['Close'], 
        start_p=1, start_q=1,
        test='adf',       # Use ADF test to find optimal 'd'
        max_p=3, max_q=3, # Maximum p and q
        m=1,              # No seasonality (m=1) for daily data
        d=None,           # Let model determine 'd'
        seasonal=False,   # No seasonal component
        start_P=0, 
        D=0, 
        trace=False,
        error_action='ignore',  
        suppress_warnings=True, 
        stepwise=True
    )
    
    # Make forecasts for the next n_periods (trading days)
    forecast_results = model.predict(n_periods=n_periods, return_conf_int=True)
    
    # Extract the predictions and confidence intervals
    predictions = forecast_results[0]
    conf_int = forecast_results[1]

    # Create a DataFrame for the forecast results
    # Generate the dates for the next n_periods
    last_date = df.index[-1].date()
    # Find the next 'n_periods' valid trading days (skipping weekends/holidays)
    future_dates = []
    current_date = last_date
    while len(future_dates) < n_periods:
        current_date += timedelta(days=1)
        # Check if it's a weekday (Monday=0 to Friday=4)
        if current_date.weekday() < 5: 
            future_dates.append(current_date)

    forecast_df = pd.DataFrame({
        'Predicted Price': predictions.values,
        'Lower Bound (95%)': conf_int[:, 0],
        'Upper Bound (95%)': conf_int[:, 1]
    }, index=future_dates)
    
    print("-> Forecast Complete.")
    return forecast_df

def determine_market_status(current_price, predicted_price):
    """Provides a simple general stock market status based on the prediction."""
    change = predicted_price - current_price
    
    if change > current_price * 0.005:  # Arbitrary threshold for "Bullish"
        status = "Bullish (Expected Increase)"
    elif change < current_price * -0.005: # Arbitrary threshold for "Bearish"
        status = "Bearish (Expected Decrease)"
    else:
        status = "Neutral (Expected Little Change)"
        
    return status, change

# --- Main Program Execution ---
def main():
    print("Stock Price Predictor (ARIMA Model)")
    print("-----------------------------------")
    
    # User Input
    ticker = input("Enter the stock ticker symbol (e.g., AAPL, MSFT, GOOGL): ").strip().upper()
    
    # 1. Fetch Data
    historical_data = fetch_historical_data(ticker)
    if historical_data is None:
        return

    # 2. Get Today's Last Known Price
    if historical_data.empty:
        print("Error: Historical data is empty.")
        return
        
    last_close_price = historical_data['Close'].iloc[-1]
    
    print("\n--- Current Data ---")
    print(f"Last available closing date: {historical_data.index[-1].strftime('%Y-%m-%d')}")
    print(f"Last closing price: ${last_close_price[0]}")

    # 3. Generate Predictions (for today and tomorrow's close)
    # The first prediction will be for the next *trading* day (which could be today or tomorrow)
    # The second prediction will be for the day after that.
    forecast_periods = 2
    forecast_results = predict_stock_price(historical_data, n_periods=forecast_periods)

    if forecast_results is None or forecast_results.empty:
        print("Error: Could not generate a forecast.")
        return

    # 4. Display Results and Market Status
    print("\n--- Price Prediction ---")
    
    # Today's Prediction (if market is currently open or just closed for the day)
    # Note: For simplicity, the first predicted value will be treated as the prediction 
    # for the next *trading day's close*, which could be 'today' if the request is late 
    # in the day, or 'tomorrow' if the request is earlier.
    today_date_str = date.today().strftime('%Y-%m-%d')
    
    # Prediction for the next trading day's close
    next_day_pred = forecast_results.iloc[0]['Predicted Price']
    next_day_date = forecast_results.index[0].strftime('%Y-%m-%d')
    status_1, change_1 = determine_market_status(last_close_price[0], next_day_pred)
    
    print(f"\nPrediction for **{next_day_date} (Next Trading Day)**:")
    print(f"  Predicted Closing Price: **${next_day_pred:.2f}**")
    print(f"  Change from last close: {change_1:+.2f}")
    print(f"  General Market Status: **{status_1}**")
    print(f"  95% Confidence Interval: [${forecast_results.iloc[0]['Lower Bound (95%)']:.2f} - ${forecast_results.iloc[0]['Upper Bound (95%)']:.2f}]")

    # Prediction for the day after that
    if forecast_periods > 1:
        day_after_pred = forecast_results.iloc[1]['Predicted Price']
        day_after_date = forecast_results.index[1].strftime('%Y-%m-%d')
        status_2, change_2 = determine_market_status(last_close_price[0], day_after_pred)

        print(f"\nPrediction for **{day_after_date} (The Trading Day After)**:")
        print(f"  Predicted Closing Price: **${day_after_pred:.2f}**")
        print(f"  Change from last close: {change_2:+.2f}")
        print(f"  General Market Status: **{status_2}**")
        print(f"  95% Confidence Interval: [${forecast_results.iloc[1]['Lower Bound (95%)']:.2f} - ${forecast_results.iloc[1]['Upper Bound (95%)']:.2f}]")

if __name__ == "__main__":
    main()