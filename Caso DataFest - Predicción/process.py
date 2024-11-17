# Import necessary libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
# For time series clustering and DTW
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import dtw
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

# For linear regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from skimpy import skim, generate_test_data
import seaborn as sns
# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
from collections import deque

from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from pandas.tseries.offsets import DateOffset
import joblib
from collections import defaultdict

from collections import defaultdict
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from pandas.tseries.offsets import DateOffset
import lightgbm as lgb
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
import joblib
import random
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from pandas.tseries.offsets import DateOffset
import joblib
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

def preprocess_atm_data(df):
    # Convert transactionTime to datetime
    df['transactionTime'] = pd.to_datetime(df['transactionTime'], format='%d/%m/%Y')
    
    # Extract date-related features
    df['day_of_week'] = df['transactionTime'].dt.dayofweek  # Monday=0, Sunday=6
    df['month'] = df['transactionTime'].dt.month
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)  # Weekend indicator
    df['day'] = df['transactionTime'].dt.day
    #df['transactionTime'] = df['transactionTime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    # Cyclic encoding for hour (0-23)
    #df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    #df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Cyclic encoding for day_of_week (0-6)
    # Rush hour feature
    def is_rush_hour(hour):
        if 1 <= hour <= 7:
            return 1
        elif 8 <= hour <= 14:
            return 2
        elif 15 <= hour <= 21:
            return 3
        elif 22 <= hour <= 31:
            return 4
    df['week'] = df['day'].apply(is_rush_hour)

    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    df['week_sin'] = np.sin(2 * np.pi * df['week'] / 4)
    df['week_cos'] = np.cos(2 * np.pi * df['week'] / 4)
    
    # Sort data by ATM ID and transaction time
    df.sort_values(['atmId', 'transactionTime'], inplace=True)
    
    # Feature Engineering: Calculate balance change to identify transactions
    #df['aux_3'] = df.groupby('atmId')['totalBalance'].diff().fillna(0)
    
    # Identify cash replenishment events (sudden increase in balance)
    df['is_replenishment'] = df['abastecimiento'].apply(lambda x: 1 if x > 0 else 0)
    
    #df['balance_change'] = df['totalIncome'] - df['totalOutcome']
    # Lag features for the past hours
    #df['balance_change_lag1'] = df.groupby('atmId')['balance_change'].shift(1)
    #df['balance_change_lag2'] = df.groupby('atmId')['balance_change'].shift(2)
    #df['balance_change_lag3'] = df.groupby('atmId')['balance_change'].shift(3)

    # Adding 24-hour lag (lag of 24 hours)
    df['balance_change_lag24'] = df.groupby('atmId')['balance_change'].shift(24)

    # Rolling mean of the past 3, 6, and 12 hours
    #df['rolling_mean_3h'] = df.groupby('atmId')['balance_change'].rolling(window=3).mean().reset_index(0, drop=True)
    #df['rolling_mean_6h'] = df.groupby('atmId')['balance_change'].rolling(window=6).mean().reset_index(0, drop=True)
    df['rolling_mean_24h'] = df.groupby('atmId')['balance_change'].rolling(window=24).mean().reset_index(0, drop=True)

    # Difference between current balance and rolling mean of 6 hours
    # Drop unnecessary columns
    df.drop(columns=['abastecimiento', 'saldo_final'], inplace=True)
    
    # Return the processed dataframe
    return df

# Call the function
def glimpse(df):
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    for col in df.columns:
        print(f"$ {col} <{df[col].dtype}> {df[col].head().values}")


def process_data(df, exclude_columns=None):
    """
    Preprocessing the data: Handle missing values, scaling, and encoding.
    """
    if exclude_columns is None:
        exclude_columns = ['atmId', 'transactionTime', 'day', 'is_replenishment', 'cluster_label']
    
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Handle missing values
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    # Ensure 'transactionTime' is in datetime format
    df['transactionTime'] = pd.to_datetime(df['transactionTime']).dt.strftime('%Y-%d-%m %H:%M:%S')
    
    # Prepare features for training and testing
    features = [col for col in df.columns if col not in exclude_columns and col != 'balance_change']
    
    return df, features



#### Extra variables not explored

def add_time_features(df):
    df['hour'] = df['transactionTime'].dt.hour
    df['day_of_week'] = df['transactionTime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['day_of_month'] = df['transactionTime'].dt.day
    df['month'] = df['transactionTime'].dt.month
    return df
def add_cyclical_features(df):
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    return df
def create_ema_features(df, spans):
    for span in spans:
        df[f'ema_{span}'] = df['balance_change'].ewm(span=span, adjust=False).mean()
    return df
def create_diff_features(df, lags):
    for lag in lags:
        df[f'diff_lag{lag}'] = df['balance_change'] - df[f'balance_change_lag{lag}']
    return df
def create_rolling_std_features(df, windows):
    for window in windows:
        df[f'rolling_std_{window}h'] = df['balance_change'].rolling(window=window).std()
    return df
def create_rolling_skew_kurt_features(df, windows):
    for window in windows:
        df[f'rolling_skew_{window}h'] = df['balance_change'].rolling(window=window).skew()
        df[f'rolling_kurt_{window}h'] = df['balance_change'].rolling(window=window).kurt()
    return df


def impute_missing_values(df, time_col='transactionTime', method='linear'):
    """
    Impute missing values in the DataFrame using the specified method.
    """
    df = df.set_index(time_col)
    df = df.asfreq('d')  # Ensure the DataFrame has hourly frequency
    
    # Interpolate missing values
    df_interpolated = df.interpolate(method=method)

    # Fill any remaining NaNs (e.g., at the start or end)
    df_interpolated = df_interpolated.fillna(method='bfill').fillna(method='ffill')

    df_interpolated = df_interpolated.reset_index()
    return df_interpolated

def columns_with_na(df):
    # Filtra las columnas que tienen al menos un valor NA
    return df.columns[df.isna().any()].tolist()

# Usar la función con tu DataFrame
#columns_with_na_list2 = columns_with_na(df_cleaned_futuro2)
#print(columns_with_na_list2)


import pandas as pd
from sklearn.impute import KNNImputer

def impute_missing_values_knn(df, columns_to_impute, id_columns, n_neighbors=3):
    """
    Función para imputar valores faltantes en un DataFrame utilizando el método KNN.
    
    Parámetros:
    df (pd.DataFrame): DataFrame que contiene los datos con valores faltantes.
    columns_to_impute (list): Lista de columnas que se van a imputar.
    id_columns (list): Lista de nombres de columnas que no se deben utilizar en la imputación.
    n_neighbors (int): Número de vecinos a considerar para la imputación (por defecto 5).
    
    Retorno:
    pd.DataFrame: DataFrame con los valores faltantes imputados.
    """
    # Crea una copia del DataFrame original
    df_copy = df.copy()

    # Separa las columnas a imputar y excluye las columnas ID
    df_features = df_copy.drop(columns=id_columns)
    
    # Solo mantiene las columnas que se desean imputar
    df_features = df_features[columns_to_impute]

    # Inicializa el imputador KNN
    imputer = KNNImputer(n_neighbors=n_neighbors)

    # Imputa los valores faltantes
    df_imputed_features = imputer.fit_transform(df_features)

    # Convierte de nuevo a DataFrame
    df_imputed_features = pd.DataFrame(df_imputed_features, columns=columns_to_impute)

    # Reemplaza las columnas imputadas en el DataFrame original
    df_copy[columns_to_impute] = df_imputed_features

    return df_copy

# Uso de la función
#columns_to_impute = columns_with_na_list
#id_columns = ['atmId',"hour"]  # Lista de columnas que no se usarán en la imputación
#df_imputed = impute_missing_values_knn(df_cleaned_futuro2, columns_with_na_list2, id_columns, n_neighbors=3)

# Revisa si quedan NA
#print(df_imputed.isna().sum())



import pandas as pd

def create_special_day_effects(df):
    """
    Función para crear columnas que identifiquen días especiales y los efectos pre y post de estos días.

    Parámetros:
    df (pd.DataFrame): DataFrame con las columnas 'transactionTime' y 'dayOfMonth'.

    Retorno:
    pd.DataFrame: DataFrame con las nuevas columnas 'is_special_day', 'pre_efecto' y 'post_efecto'.
    """
    # Convertir 'transactionTime' a tipo datetime si no lo está
    df['transactionTime'] = pd.to_datetime(df['transactionTime'])

    # Crear una nueva columna binaria para los días especiales (1, 24, 30)
    df['is_special_day'] = df['day'].apply(lambda x: 1 if x in [1, 15,25, 30] else 0)

    # Inicializar columnas para Pre-efecto y Post-efecto
    df['pre_efecto'] = 0
    df['post_efecto'] = 0

    # Recorrer el DataFrame para marcar los días pre y post efecto
    for index in range(1, len(df)):
        if df['is_special_day'].iloc[index] == 1:  # Si es un día especial
            # Marcar el día anterior como pre-efecto (si no es el primer día)
            if index > 0:
                df.at[index - 1, 'pre_efecto'] = 1
            # Marcar el día siguiente como post-efecto (si no es el último día)
            if index < len(df) - 1:
                df.at[index + 1, 'post_efecto'] = 1

    return df

# Supongamos que ya tienes el DataFrame df con las columnas 'transactionTime' y 'dayOfMonth'
# Uso de la función
#df_cleaned = create_special_day_effects(df_cleaned)

# Mostrar las primeras filas del DataFrame para verificar
#print(df_cleaned[['transactionTime', 'dayOfMonth', 'is_special_day', 'pre_efecto', 'post_efecto']].head(10))
