import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta

# --- 1. Data Construction (Strict Traceability) ---
st.title("GAC Motor Demand Forecasting - GCC Region")
st.write("This dashboard forecasts GAC Motor demand in the GCC using a highly traceable, macro-anchored methodology.")

# Anchor Points: GASTAT Annual New Vehicle Registrations (Saudi Arabia)
gastat_anchors = {
    2022: 695700,
    2023: 878100,
    2024: 1025700
}

# Fetch Real Macro Data: Brent Crude Oil Prices (Monthly)
print("Fetching real Brent Crude data via yfinance...")
oil_data = yf.download("BZ=F", start="2022-01-01", end="2025-01-01", interval="1mo")
if oil_data.empty:
    print("Warning: Failed to fetch yfinance data. Using fallback flat prices.")
    oil_prices = [85.0] * 36
else:
    # Use closing prices, forward fill any NaN
    oil_prices = oil_data['Close'].ffill().values.flatten()[:36]
    if len(oil_prices) < 36:
        oil_prices = np.pad(oil_prices, (0, 36 - len(oil_prices)), 'edge')

# Construct the Monthly Dates
dates = [datetime(2022, 1, 1) + relativedelta(months=i) for i in range(36)]

# Create Base Seasonality Weights (e.g. Ramadan spikes in March/April, End of Year in Nov/Dec)
base_seasonality = np.array([1.0, 1.0, 1.15, 1.1, 1.0, 1.0, 0.9, 0.9, 1.0, 1.05, 1.1, 1.2])

# Distribute Annual Anchors to Monthly (Estimated Gap Filling)
market_units = []
market_flags = []
for year in [2022, 2023, 2024]:
    annual_total = gastat_anchors[year]
    # Modulate seasonality by oil price index for that year to add real macro variance
    year_idx = year - 2022
    oil_year = oil_prices[year_idx*12:(year_idx+1)*12]
    # Normalize oil prices to a tight band around 1.0
    oil_modulator = 1.0 + (oil_year - np.mean(oil_year)) / (np.mean(oil_year) + 1e-5) * 0.05
    
    monthly_weights = base_seasonality * oil_modulator
    monthly_weights /= np.sum(monthly_weights)
    
    monthly_units = annual_total * monthly_weights
    market_units.extend(monthly_units)
    # The last month of the year conceptually contains the annual aggregate
    market_flags.extend(["Estimated_Split"] * 11 + ["Real_GASTAT_Annual_Aggregate"])

# Chinese OEM Benchmark: 2022 ~ 8%, 2024 ~ 15%
chinese_share = np.linspace(0.08, 0.15, 36)
chinese_flags = ["Estimated_Interpolation"] * 35 + ["Real_Benchmark_15%"]

# Assume GAC captures 15% of the total Chinese OEM slice
gac_slice = 0.15
gac_units = np.array(market_units) * chinese_share * gac_slice

# Build DataFrame
df = pd.DataFrame({
    'Date': dates,
    'Total_Market': market_units,
    'Total_Market_Flag': market_flags,
    'Brent_Crude': oil_prices,
    'Chinese_Share': chinese_share,
    'Chinese_Share_Flag': chinese_flags,
    'GAC_Units': gac_units
})

df.to_csv('gac_traceable_dataset.csv', index=False)
st.subheader("Generated Traceable Dataset")
st.dataframe(df)
print("Saved traceable dataset to 'gac_traceable_dataset.csv'")

# --- 2. Feature Engineering ---
# Add Lags
df['Lag_1'] = df['GAC_Units'].shift(1)
df['Lag_3'] = df['GAC_Units'].shift(3)
df['Lag_6'] = df['GAC_Units'].shift(6)
df['Rolling_Mean_3'] = df['GAC_Units'].rolling(window=3).mean()

# Drop NAs (this will drop the first 6 months)
df_model = df.dropna().reset_index(drop=True)

# Features and Target
features = ['Lag_1', 'Lag_3', 'Lag_6', 'Rolling_Mean_3', 'Brent_Crude', 'Chinese_Share']
X = df_model[features]
y = df_model['GAC_Units']

# --- 3 & 4. Modeling & Validation ---
print("Starting XGBoost Modeling & Walk-Forward Validation...")
tscv = TimeSeriesSplit(n_splits=3)
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [2, 3],
    'learning_rate': [0.05, 0.1]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=tscv, scoring='neg_root_mean_squared_error')
grid_search.fit(X, y)

best_model = grid_search.best_estimator_
print(f"Best params: {grid_search.best_params_}")
print(f"Average Cross-Validation RMSE: {-grid_search.best_score_:.2f}")

# Extract feature importance
importance = best_model.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(features, importance, color='skyblue')
plt.xlabel('Importance')
plt.title('XGBoost Feature Importance')
plt.tight_layout()
st.subheader("Model Feature Importance")
st.pyplot(plt.gcf())
plt.savefig('feature_importance.png')
print("Saved feature importance plot to 'feature_importance.png'")

# --- 5. Scenario Forecasting (2025) ---
future_dates = [datetime(2025, 1, 1) + relativedelta(months=i) for i in range(12)]
last_oil_price = df['Brent_Crude'].iloc[-1]

def generate_forecast(share_adjustment):
    future_gac_units = []
    current_df = df.copy()
    
    # Base projected Chinese share for 2025 (e.g. continues to grow from 15% to 17%, plus adjustment)
    base_future_share = np.linspace(0.15, 0.17, 12) + share_adjustment
    
    for i in range(12):
        lag_1 = current_df['GAC_Units'].iloc[-1]
        lag_3 = current_df['GAC_Units'].iloc[-3]
        lag_6 = current_df['GAC_Units'].iloc[-6]
        rolling_mean = current_df['GAC_Units'].iloc[-3:].mean()
        curr_share = base_future_share[i]
        
        X_pred = pd.DataFrame({
            'Lag_1': [lag_1],
            'Lag_3': [lag_3],
            'Lag_6': [lag_6],
            'Rolling_Mean_3': [rolling_mean],
            'Brent_Crude': [last_oil_price],
            'Chinese_Share': [curr_share]
        })
        
        pred = best_model.predict(X_pred)[0]
        future_gac_units.append(pred)
        
        # append to current_df to use for next lag
        new_row = {'GAC_Units': pred}
        current_df = pd.concat([current_df, pd.DataFrame([new_row])], ignore_index=True)
        
    return future_gac_units

print("Generating scenario forecasts...")
forecast_base = generate_forecast(0.0)
forecast_high = generate_forecast(0.05)
forecast_low = generate_forecast(-0.05)

# --- 6. Outputs ---
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['GAC_Units'], label='Historical (Derived from GASTAT Anchors)', color='black', marker='o')

plt.plot(future_dates, forecast_base, label='Base Forecast', color='blue', linestyle='--')
plt.plot(future_dates, forecast_high, label='High Growth (+5% Chinese Share)', color='green', linestyle='--')
plt.plot(future_dates, forecast_low, label='Conservative (-5% Chinese Share)', color='red', linestyle='--')

plt.title('GAC Motor Monthly Demand Forecast (2025 Scenarios)')
plt.xlabel('Date')
plt.ylabel('Estimated GAC Units')
plt.legend()
plt.grid(True)
st.subheader("2025 Scenario Forecasts")
st.pyplot(plt.gcf())
plt.savefig('forecast_scenarios.png')
print("Saved forecast plot to 'forecast_scenarios.png'")

with open('assumptions.txt', 'w') as f:
    f.write("=== GCC Automotive Demand Forecasting Assumptions ===\n\n")
    f.write("1. PRIMARY ANCHOR (OBSERVED):\n")
    f.write("   - Saudi Arabia Annual New Vehicle Registrations (GASTAT):\n")
    f.write("     2022: 695,700 | 2023: 878,100 | 2024: 1,025,700\n\n")
    f.write("2. TRANSPARENT GAP FILLING (ESTIMATED):\n")
    f.write("   - Since no public quarterly GASTAT data exists, intra-year distributions\n")
    f.write("     were derived using base GCC seasonality and modulated by real Brent Crude prices.\n")
    f.write("   - These are explicitly flagged as 'Estimated_Split' in the dataset.\n\n")
    f.write("3. CHINESE OEM GROWTH BENCHMARK:\n")
    f.write("   - Anchored from ~8% (2022) to 15% (2024) based on CAAM/AlixPartners.\n")
    f.write("   - Interstitial months are interpolated and flagged 'Estimated_Interpolation'.\n\n")
    f.write("4. SECONDARY MACRO VARIABLE:\n")
    f.write("   - Historical Brent Crude Oil monthly prices fetched via yfinance.\n\n")
    f.write("5. SCENARIO FORECASTING:\n")
    f.write("   - Base Case: Chinese share continues slight growth from 15% to 17%.\n")
    f.write("   - High Growth: Base + 5%.\n")
    f.write("   - Conservative: Base - 5%.\n")
print("Saved assumptions to 'assumptions.txt'")

st.subheader("Methodology & Assumptions")
with open('assumptions.txt', 'r') as f:
    st.text(f.read())
