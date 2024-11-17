import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.stattools import acf, pacf
from scipy.stats import norm
import numpy as np
 

##OVERALL FUNCTION UNIVARIATE SERIES
def analyze_univariate_time_series(atm_data, atm_id):
    """
    Perform time series analysis for a univariate series by calling individual analysis functions.

    Parameters:
    - atm_data: DataFrame containing 'transactionTime' and 'balance_change' (and possibly other variables).
    - atm_id: Identifier for the ATM (for labeling plots).
    """
    # Data preparation
    atm_data_prepared = prepare_time_series_data(atm_data)

    # Time series decomposition
    perform_time_series_decomposition(atm_data_prepared, atm_id)

    # ADF test
    adf_result = perform_adf_test(atm_data_prepared)
    print('ADF Statistic:', adf_result['ADF Statistic'])
    print('p-value:', adf_result['p-value'])
    for key, value in adf_result['Critical Values'].items():
        print(f'Critical Value ({key}): {value}')
    print(f"\nConclusion: The time series is {adf_result['Conclusion']}.")

    # Plot ACF
    plot_acf_function(atm_data_prepared, atm_id)

    # Plot PACF
    plot_pacf_function(atm_data_prepared, atm_id)

    # Plot daily seasonality
    plot_daily_seasonality(atm_data_prepared, atm_id)

    # Plot weekly seasonality
    plot_weekly_seasonality(atm_data_prepared, atm_id)

    # Plot box plot to identify outliers
    plot_boxplot_outliers(atm_data_prepared, atm_id)

    # Plot time series to spot anomalies
    plot_time_series_anomalies(atm_data_prepared, atm_id)

    # Plot distribution of balance change
    plot_balance_change_distribution(atm_data_prepared, atm_id)

    # Compute and display correlation matrix
    compute_correlation_matrix(atm_data_prepared, atm_id)





def prepare_time_series_data(atm_data):
    """
    Prepare the time series data by ensuring datetime format, setting index,
    and interpolating missing values.

    Parameters:
    - atm_data: DataFrame containing 'transactionTime' and 'balance_change'.

    Returns:
    - atm_data_prepared: Prepared DataFrame with 'transactionTime' as index.
    """
    atm_data_prepared = atm_data.copy()
    atm_data_prepared['transactionTime'] = pd.to_datetime(atm_data_prepared['transactionTime'])
    atm_data_prepared.set_index('transactionTime', inplace=True)
    atm_data_prepared.sort_index(inplace=True)
    # Fill missing values if any
    atm_data_prepared['balance_change'].interpolate(method='time', inplace=True)
    return atm_data_prepared

def perform_time_series_decomposition(atm_data, atm_id, model='additive', period=24):
    """
    Perform time series decomposition and plot the components.

    Parameters:
    - atm_data: DataFrame with 'balance_change' and 'transactionTime' as index.
    - atm_id: Identifier for the ATM (for labeling plots).
    - model: Type of decomposition ('additive' or 'multiplicative').
    - period: Period for seasonal decomposition (e.g., 24 for hourly data with daily seasonality).
    """
    decomposition = sm.tsa.seasonal_decompose(atm_data['balance_change'], model=model, period=period)

    # Plot the decomposition
    fig = decomposition.plot()
    fig.set_size_inches(12, 8)
    fig.suptitle(f'Time Series Decomposition for ATM {atm_id}', fontsize=16)
    plt.show()



def perform_adf_test(atm_data):
    """
    Perform the Augmented Dickey-Fuller test for stationarity.

    Parameters:
    - atm_data: DataFrame with 'balance_change' column.

    Returns:
    - result_dict: Dictionary containing ADF statistic, p-value, and critical values.
    """
    result = adfuller(atm_data['balance_change'].dropna())
    result_dict = {
        'ADF Statistic': result[0],
        'p-value': result[1],
        'Critical Values': result[4],
        'Conclusion': 'Stationary' if result[1] < 0.05 else 'Non-Stationary'
    }
    return result_dict

def plot_acf_function(atm_data, atm_id, lags=50):
    """
    Plot the Autocorrelation Function (ACF).

    Parameters:
    - atm_data: DataFrame with 'balance_change' column.
    - atm_id: Identifier for the ATM (for labeling plots).
    - lags: Number of lags to include in the plot.
    """
    plt.figure(figsize=(10, 5))
    plot_acf(atm_data['balance_change'].dropna(), lags=lags, ax=plt.gca())
    plt.title(f'Autocorrelation Function (ACF) for ATM {atm_id}')
    plt.xlabel('Lags')
    plt.ylabel('Autocorrelation')
    plt.show()

def plot_pacf_function(atm_data, atm_id, lags=50):
    """
    Plot the Partial Autocorrelation Function (PACF).

    Parameters:
    - atm_data: DataFrame with 'balance_change' column.
    - atm_id: Identifier for the ATM (for labeling plots).
    - lags: Number of lags to include in the plot.
    """
    plt.figure(figsize=(10, 5))
    plot_pacf(atm_data['balance_change'].dropna(), lags=lags, ax=plt.gca(), method='ywm')
    plt.title(f'Partial Autocorrelation Function (PACF) for ATM {atm_id}')
    plt.xlabel('Lags')
    plt.ylabel('Partial Autocorrelation')
    plt.show()


def compute_correlation_matrix(atm_data, atm_id):
    """
    Compute and plot the correlation matrix for all numeric variables.

    Parameters:
    - atm_data: DataFrame containing numeric variables.
    - atm_id: Identifier for the ATM (for labeling plots).
    """
    numeric_cols = atm_data.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 1:
        corr_matrix = atm_data[numeric_cols].corr()
        print("\nCorrelation Matrix:")
        print(corr_matrix)
        # Plot the correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title(f'Correlation Matrix for ATM {atm_id}')
        plt.show()
    else:
        print("\nNo additional numeric variables to compute correlation matrix.")


def plot_daily_seasonality(atm_data, atm_id):
    """
    Plot the average balance change by hour to observe daily seasonality.

    Parameters:
    - atm_data: DataFrame with 'balance_change' and 'transactionTime' as index.
    - atm_id: Identifier for the ATM (for labeling plots).
    """
    atm_data['hour'] = atm_data.index.hour
    hourly_mean = atm_data.groupby('hour')['balance_change'].mean()

    plt.figure(figsize=(10, 4))
    plt.plot(hourly_mean.index, hourly_mean.values, marker='o')
    plt.title(f'Average Balance Change by Hour for ATM {atm_id}')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Balance Change')
    plt.xticks(range(0, 24))
    plt.grid(True)
    plt.show()

def plot_weekly_seasonality(atm_data, atm_id):
    """
    Plot the average balance change by day of the week to observe weekly seasonality.

    Parameters:
    - atm_data: DataFrame with 'balance_change' and 'transactionTime' as index.
    - atm_id: Identifier for the ATM (for labeling plots).
    """
    atm_data['day_of_week'] = atm_data.index.dayofweek  # Monday=0, Sunday=6
    daily_mean = atm_data.groupby('day_of_week')['balance_change'].mean()

    plt.figure(figsize=(10, 4))
    plt.plot(daily_mean.index, daily_mean.values, marker='o')
    plt.title(f'Average Balance Change by Day of Week for ATM {atm_id}')
    plt.xlabel('Day of Week (0=Monday)')
    plt.ylabel('Average Balance Change')
    plt.xticks(range(0, 7))
    plt.grid(True)
    plt.show()
def plot_boxplot_outliers(atm_data, atm_id):
    """
    Plot a box plot to identify outliers in the balance change data.

    Parameters:
    - atm_data: DataFrame with 'balance_change' column.
    - atm_id: Identifier for the ATM (for labeling plots).
    """
    plt.figure(figsize=(8, 6))
    plt.boxplot(atm_data['balance_change'].dropna(), vert=False)
    plt.title(f'Box Plot of Balance Change for ATM {atm_id}')
    plt.xlabel('Balance Change')
    plt.show()
def plot_time_series_anomalies(atm_data, atm_id):
    """
    Plot the balance change over time to spot anomalies.

    Parameters:
    - atm_data: DataFrame with 'balance_change' and 'transactionTime' as index.
    - atm_id: Identifier for the ATM (for labeling plots).
    """
    plt.figure(figsize=(15, 6))
    plt.plot(atm_data.index, atm_data['balance_change'], label='Balance Change')
    plt.title(f'Balance Change Over Time for ATM {atm_id}')
    plt.xlabel('Transaction Time')
    plt.ylabel('Balance Change')
    plt.legend()
    plt.grid(True)
    plt.show()
def plot_balance_change_distribution(atm_data, atm_id):
    """
    Plot the distribution of balance change using a histogram with KDE.

    Parameters:
    - atm_data: DataFrame with 'balance_change' column.
    - atm_id: Identifier for the ATM (for labeling plots).
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(atm_data['balance_change'].dropna(), bins=50, kde=True)
    plt.title(f'Distribution of Balance Change for ATM {atm_id}')
    plt.xlabel('Balance Change')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
###########################################################
###########################################################
### Lags and rolling mean
def process_lag_data(atm_df):
    # Set 'transactionTime' as the index
    atm_df = atm_df.sort_values('transactionTime')
    atm_df.set_index('transactionTime', inplace=True)

    # Get significant lags
    significant_lags = get_significant_lags(atm_df['balance_change'])

    # Create lag features
    atm_df = create_lag_features(atm_df, significant_lags)

    # Reset index
    atm_df.reset_index(inplace=True)
    return atm_df, significant_lags

def plot_acf_pacf(atm_data, atm_id, lags=50):
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plot_acf(atm_data['balance_change'].dropna(), lags=lags, ax=plt.gca())
    plt.title(f'ACF for ATM {atm_id}')
    plt.subplot(122)
    plot_pacf(atm_data['balance_change'].dropna(), lags=lags, ax=plt.gca())
    plt.title(f'PACF for ATM {atm_id}')
    plt.tight_layout()
    plt.show()



def create_lag_features(df, lags):
    for lag in lags:
        df[f'balance_change_lag{lag}'] = df['balance_change'].shift(lag)
    return df
def create_rolling_mean_features(df, windows):
    for window in windows:
        df[f'rolling_mean_{window}h'] = df['balance_change'].rolling(window=window).mean()
    return df
def plot_rolling_means(atm_data, atm_id, windows=[3, 6, 12, 24]):
    plt.figure(figsize=(15, 7))
    plt.plot(atm_data['transactionTime'], atm_data['balance_change'], label='Original')

    for window in windows:
        atm_data[f'rolling_mean_{window}h'] = atm_data['balance_change'].rolling(window=window).mean()
        plt.plot(atm_data['transactionTime'], atm_data[f'rolling_mean_{window}h'], label=f'Rolling Mean {window}h')

    plt.title(f'Rolling Means for ATM {atm_id}')
    plt.xlabel('Transaction Time')
    plt.ylabel('Balance Change')
    plt.legend()
    plt.show()


def get_significant_lags(series, max_lag=158, alpha=0.001):
    """
    Identify significant lags based on ACF and PACF values.
    """
    N = len(series)
    z = norm.ppf(1 - alpha / 2)
    threshold = z / np.sqrt(N)
    
    acf_vals = acf(series, nlags=max_lag, fft=False)
    #pacf_vals = pacf(series, nlags=max_lag, method='ywm')
    
    significant_lags = []
    for lag in range(1, len(acf_vals)):
        if abs(acf_vals[lag]) > threshold:
            significant_lags.append(lag)
    return significant_lags

def plot_rolling_means(atm_data, atm_id, windows=[3, 6, 12, 24]):
    plt.figure(figsize=(15, 7))
    plt.plot(atm_data['transactionTime'], atm_data['balance_change'], label='Original')

    for window in windows:
        atm_data[f'rolling_mean_{window}h'] = atm_data['balance_change'].rolling(window=window).mean()
        plt.plot(atm_data['transactionTime'], atm_data[f'rolling_mean_{window}h'], label=f'Rolling Mean {window}h')

    plt.title(f'Rolling Means for ATM {atm_id}')
    plt.xlabel('Transaction Time')
    plt.ylabel('Balance Change')
    plt.legend()
    plt.show()

