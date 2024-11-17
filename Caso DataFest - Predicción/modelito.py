# Import necessary libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

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
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from pandas.tseries.offsets import DateOffset
import joblib
from collections import defaultdict
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from pandas.tseries.offsets import DateOffset
import lightgbm as lgb
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
import joblib

import pandas as pd
import numpy as np
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

import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
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
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from scipy.stats import uniform, randint
import lightgbm as lgb



def evaluate_model(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return rmse, mape


def train_lightgbm(train_df, test_df, features, target,n_splits=2):
    import lightgbm as lgb

    X_train = train_df[features].fillna(0)
    y_train = train_df[target]
    X_test = test_df[features].fillna(0)
    y_test = test_df[target]

    # Define the parameter distribution for RandomizedSearchCV
    param_dist = {
        'objective': ['regression'],
        'metric': ['rmse'],
        'learning_rate': uniform(0.01, 0.3),
        'num_leaves': randint(16, 64),
        'max_depth': randint(3, 7),
        'n_estimators': randint(50, 200),
        'lambda_l1': uniform(0, 10),
        'lambda_l2': uniform(0, 10)
    }

    lgbm_model = lgb.LGBMRegressor()

    # Hyperparameter tuning
    best_model, best_params = hyperparameter_tuning(lgbm_model, param_dist, X_train, y_train, n_iter=7, n_splits=n_splits)
    print(f"Best parameters for LightGBM: {best_params}")

    # Rolling Cross-Validation
    avg_rmse_cv, avg_mape_cv, _ = rolling_cross_validation(best_model, X_train, y_train, n_splits=n_splits)

    # Fit on full training data
    #best_model.fit(X_train, y_train)
    best_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(stopping_rounds=10)])
    #model = lgb.train(params, train_data, valid_sets=[test_data], callbacks=[lgb.early_stopping(stopping_rounds=10)])

    # Evaluate on test data
    y_pred = best_model.predict(X_test)
    rmse, mape = evaluate_model(y_test, y_pred)

    return avg_rmse_cv, avg_mape_cv, y_pred.tolist(), best_params, best_model




def train_xgboost(train_df, test_df, features, n_splits=5):
    from xgboost import XGBRegressor

    X_train = train_df[features].fillna(0)
    y_train = train_df['balance_change']
    X_test = test_df[features].fillna(0)
    y_test = test_df['balance_change']

    # Define the parameter distribution for RandomizedSearchCV
    param_dist = {
        'n_estimators': randint(50, 200),
        'max_depth': randint(3, 15),
        'learning_rate': uniform(0.01, 0.2),
        'reg_lambda': uniform(0, 10),
        'reg_alpha': uniform(0, 10),
        'subsample': uniform(0.5, 0.5),
        'colsample_bytree': uniform(0.5, 0.5)
    }

    xgb_model = XGBRegressor()

    # Hyperparameter tuning
    best_model, best_params = hyperparameter_tuning(xgb_model, param_dist, X_train, y_train, n_iter=50, n_splits=n_splits)
    print(f"Best parameters for XGBoost: {best_params}")

    # Rolling Cross-Validation
    avg_rmse_cv, avg_mape_cv, _ = rolling_cross_validation(best_model, X_train, y_train, n_splits=n_splits)

    # Fit on full training data
    best_model.fit(X_train, y_train)

    # Evaluate on test data
    y_pred = best_model.predict(X_test)
    rmse, mape = evaluate_model(y_test, y_pred)

    return avg_rmse_cv, avg_mape_cv, y_pred.tolist(), best_params, best_model



def train_catboost(train_df, test_df, features, n_splits=5):
    from catboost import CatBoostRegressor

    X_train = train_df[features].fillna(0)
    y_train = train_df['balance_change']
    X_test = test_df[features].fillna(0)
    y_test = test_df['balance_change']

    # Define the parameter distribution for RandomizedSearchCV
    param_dist = {
        'depth': randint(4, 10),
        'learning_rate': uniform(0.01, 0.3),
        'l2_leaf_reg': uniform(1, 10),
        'iterations': randint(50, 200)
    }

    cat_model = CatBoostRegressor(silent=True)

    # Hyperparameter tuning
    best_model, best_params = hyperparameter_tuning(cat_model, param_dist, X_train, y_train, n_iter=20, n_splits=n_splits)
    print(f"Best parameters for CatBoost: {best_params}")

    # Rolling Cross-Validation
    avg_rmse_cv, avg_mape_cv, _ = rolling_cross_validation(best_model, X_train, y_train, n_splits=n_splits)

    # Fit on full training data
    best_model.fit(X_train, y_train)

    # Evaluate on test data
    y_pred = best_model.predict(X_test)
    rmse, mape = evaluate_model(y_test, y_pred)

    return avg_rmse_cv, avg_mape_cv, y_pred.tolist(), best_params, best_model


def train_adaboost(train_df, test_df, features,target, n_splits=2):
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
    from scipy.stats import uniform, randint

    X_train = train_df[features].fillna(0)
    y_train = train_df[target]
    X_test = test_df[features].fillna(0)
    y_test = test_df[target]

    # Define the parameter distribution for RandomizedSearchCV
    param_dist = {
        'n_estimators': randint(50, 200),
        'learning_rate': uniform(0.01, 1.0),
        'loss': ['linear', 'square', 'exponential']
    }

    ada_model = AdaBoostRegressor(estimator=DecisionTreeRegressor())

    # Hyperparameter tuning
    best_model, best_params = hyperparameter_tuning(ada_model, param_dist, X_train, y_train, n_iter=2, n_splits=n_splits)
    print(f"Best parameters for AdaBoost: {best_params}")

    # Rolling Cross-Validation
    avg_rmse_cv, avg_mape_cv, _ = rolling_cross_validation(best_model, X_train, y_train, n_splits=n_splits)

    # Fit on full training data
    best_model.fit(X_train, y_train)

    # Evaluate on test data
    y_pred = best_model.predict(X_test)
    rmse, mape = evaluate_model(y_test, y_pred)

    return avg_rmse_cv, avg_mape_cv, y_pred.tolist(), best_params, best_model


def train_stacking_ensemble(train_df, test_df, features, base_models, meta_model):
    """
    Train a stacking ensemble model.

    Parameters:
    - train_df: Training DataFrame.
    - test_df: Testing DataFrame.
    - features: List of feature column names.
    - base_models: List of base models.
    - meta_model: Meta-model for stacking.

    Returns:
    - rmse: RMSE on the test set.
    - mape: MAPE on the test set.
    - y_pred: Predictions on the test set.
    - stacking_model: Trained stacking ensemble model.
    """
    from sklearn.ensemble import StackingRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

    X_train = train_df[features]
    y_train = train_df['balance_change']
    X_test = test_df[features]
    y_test = test_df['balance_change']

    # Handle missing values
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # Define the stacking ensemble
    estimators = [(f'model_{i}', model) for i, model in enumerate(base_models)]
    stacking_model = StackingRegressor(estimators=estimators, final_estimator=meta_model, passthrough=False, cv=5, n_jobs=-1)

    # Fit the model
    stacking_model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = stacking_model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    return rmse, mape, y_pred, stacking_model



def averaging_ensemble_predictions(predictions_list):
    """
    Compute the average of predictions from multiple models.

    Parameters:
    - predictions_list: List of prediction arrays.

    Returns:
    - averaged_predictions: Averaged prediction array.
    """
    import numpy as np
    averaged_predictions = np.mean(predictions_list, axis=0)
    return averaged_predictions


def hyperparameter_tuning(model, param_distributions, X_train, y_train, n_iter=2, n_splits=2):
    from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

    tscv = TimeSeriesSplit(n_splits=n_splits)

    randomized_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring='neg_mean_absolute_error',
        cv=tscv,
        verbose=0,
        n_jobs=-1
    )

    randomized_search.fit(X_train, y_train)

    best_model = randomized_search.best_estimator_
    best_params = randomized_search.best_params_

    return best_model, best_params




def rolling_cross_validation(model, X, y, n_splits=2, window_size=None):
    from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
    import numpy as np

    cv_results = {'RMSE': [], 'MAPE': []}
    total_samples = X.shape[0]
    fold_size = total_samples // (n_splits + 1)

    for i in range(n_splits):
        train_end = (i + 1) * fold_size
        test_end = train_end + fold_size

        if window_size:
            train_start = max(0, train_end - window_size)
        else:
            train_start = 0  # Expanding window

        X_train = X.iloc[train_start:train_end].fillna(0)
        y_train = y.iloc[train_start:train_end]
        X_test = X.iloc[train_end:test_end].fillna(0)
        y_test = y.iloc[train_end:test_end]

        # Fit the model
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        cv_results['RMSE'].append(rmse)
        cv_results['MAPE'].append(mape)

    # Average RMSE and MAPE over folds
    avg_rmse = np.mean(cv_results['RMSE'])
    avg_mape = np.mean(cv_results['MAPE'])

    # Return the last fitted model
    return avg_rmse, avg_mape, model

def train_ets(train_df, test_df):
    """
    Train an ETS model.
    """
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    y_train = train_df['balance_change']
    y_test = test_df['balance_change']

    # Fit the model
    ets_model = ETSModel(y_train, error='add', trend='add', seasonal='add', seasonal_periods=24)
    ets_fit = ets_model.fit(optimized=True)

    # Forecast
    y_pred = ets_fit.forecast(steps=len(y_test))

    # Evaluate the model
    rmse, mape = evaluate_model(y_test, y_pred)
    
    return rmse, mape, y_pred.tolist(), ets_fit



def train_arima_model(train_df, test_df):
    """
    Train an ARIMA model.
    """
    train_df = train_df.sort_values(by='transactionTime')
    test_df = test_df.sort_values(by='transactionTime')

    # Fit the ARIMA model
    arima_model = ARIMA(train_df['balance_change'], order=(1, 0, 0), seasonal_order=(0, 0, 0, 24))
    model = arima_model.fit()  # Assign the fitted model to model

    # Make predictions for the test set
    y_pred_arima = model.forecast(steps=len(test_df))
    y_test = test_df['balance_change'].values

    rmse, mape = evaluate_model(y_test, y_pred_arima)

    return rmse, mape, y_pred_arima.tolist(), model


def prepare_lstm_data(data, features, target='balance_change', time_steps=1):
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np

    # Scale features
    scaler = MinMaxScaler()
    data_scaled = data.copy()
    data_scaled[features] = scaler.fit_transform(data[features])

    X_list = []
    y_list = []

    for i in range(len(data_scaled) - time_steps):
        X_list.append(data_scaled[features].iloc[i:i+time_steps].values)
        y_list.append(data_scaled[target].iloc[i+time_steps])

    X = np.array(X_list)
    y = np.array(y_list)

    return X, y, scaler

def train_lstm(train_df, test_df, features, target='balance_change', n_splits=5, time_steps=24, epochs=10, batch_size=32):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
    import numpy as np

    # Initialize scaler
    scaler = MinMaxScaler()
    scaler.fit(train_df[features])

    X_train, y_train, _ = prepare_lstm_data(train_df, features, target, time_steps)
    X_test, y_test, _ = prepare_lstm_data(test_df, features, target, time_steps)

    # Reshape for LSTM
    X_train = X_train.reshape((X_train.shape[0], time_steps, len(features)))
    X_test = X_test.reshape((X_test.shape[0], time_steps, len(features)))

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(time_steps, len(features))))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Fit the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # Predict and inverse transform
    y_pred = model.predict(X_test).flatten()
    y_pred_rescaled = scaler.inverse_transform(
        np.hstack((test_df[features].iloc[time_steps:].values, y_pred.reshape(-1, 1)))
    )[:, -1]
    y_test_rescaled = scaler.inverse_transform(
        np.hstack((test_df[features].iloc[time_steps:].values, y_test.reshape(-1, 1)))
    )[:, -1]

    # Evaluate
    rmse = mean_squared_error(y_test_rescaled, y_pred_rescaled, squared=False)
    mape = mean_absolute_percentage_error(y_test_rescaled, y_pred_rescaled)

    return rmse, mape, y_pred_rescaled.tolist(), model, scaler



def rolling_predictions_with_lag_update(model, model_type, test_df, feature_columns, n_hours=28):
    """
    Function for rolling predictions for each ATM and updating lagged features.
    Starts with test_df and extends predictions for n_hours beyond the test data.

    Parameters:
    - model: Trained model object with a predict method.
    - model_type: String indicating the model type ('lgbm', 'xgboost', 'catboost', 'adaboost', 'ets', 'arima').
    - test_df: DataFrame containing the test data to start predictions from.
    - feature_columns: List of feature column names used for prediction.
    - n_hours: Number of hours to predict into the future.

    Returns:
    - future_predictions: List of predicted 'balance_change' values for the next n_hours.
    - future_df: DataFrame containing the predicted rows.
    """
    from collections import deque
    import numpy as np
    import pandas as pd

    future_predictions = []
    prediction_rows = []

    if model_type in ['lgbm', 'xgboost', 'catboost', 'adaboost']:
        # Ensure 'transactionTime' is datetime
        test_df = test_df.copy()
        test_df['transactionTime'] = pd.to_datetime(test_df['transactionTime'])

        # Sort by 'transactionTime'
        test_df = test_df.sort_values('transactionTime').reset_index(drop=True)

        # Initialize lag queue with actual balance_change from test_df
        lag_queue = deque([row['balance_change'] for _, row in test_df.iterrows()], maxlen=24)

        last_transaction_time = test_df['transactionTime'].iloc[-1]

        for i in range(n_hours):
            # Create new row based on last known data
            new_row = test_df.iloc[-1].copy()

            # Update 'transactionTime'
            new_row['transactionTime'] = last_transaction_time + pd.Timedelta(days=1)
            last_transaction_time = new_row['transactionTime']

            # Update time-based features
            new_row['day_of_week'] = new_row['transactionTime'].dayofweek
            new_row['day'] = new_row['transactionTime'].day
            new_row['month'] = new_row['transactionTime'].month
            new_row['is_weekend'] = 1 if new_row['day_of_week'] >= 5 else 0

            # Cyclic encoding for hour and day_of_week
            new_row['day_sin'] = np.sin(2 * np.pi * new_row['day_of_week'] / 7)
            new_row['day_cos'] = np.cos(2 * np.pi * new_row['day_of_week'] / 7)
            #new_row['week_sin'] = np.sin(2 * np.pi * new_row['week_sin'] / 4)
            #new_row['week_cos'] = np.cos(2 * np.pi * new_row['week_cos'] / 4)
            #new_row['is_special_day'] = new_row['day'].apply(lambda x: 1 if x in [1, 15,25, 30] else 0)

            # Update lag features using lag_queue
            lag_features = [col for col in test_df.columns if 'lag' in col]
            for lag_feature in lag_features:
                lag_number = int(lag_feature.replace('balance_change_lag', ''))
                if lag_number == 1:
                    new_row[lag_feature] = lag_queue[-1]
                else:
                    prev_lag = f'balance_change_lag{lag_number}'
                    new_row[lag_feature] = test_df.iloc[-lag_number][prev_lag] if lag_number <= len(test_df) else 0

            # Update rolling mean features based on previous predictions
            rolling_windows = [int(col.replace('rolling_mean_', '').replace('h', '')) for col in test_df.columns if 'rolling_mean' in col]
            for window in rolling_windows:
                window_feature_name = f'rolling_mean_{window}h'
                if window <= len(lag_queue):
                    rolling_values = list(lag_queue)[-window:]
                else:
                    rolling_values = list(lag_queue)
                new_row[window_feature_name] = np.mean(rolling_values) if len(rolling_values) > 0 else 0

            # Prepare features for prediction
            X_new = new_row[feature_columns].values.reshape(1, -1)

            # Predict
            current_prediction = model.predict(X_new)[0]
            new_row['balance_change'] = current_prediction
            new_row['is_predicted'] = True  # Mark as predicted

            # Append prediction and row
            future_predictions.append(current_prediction)
            prediction_rows.append(new_row.copy())
            lag_queue.append(current_prediction)

            # Append new_row to test_df for subsequent predictions using pd.concat
            new_row_df = pd.DataFrame([new_row])
            test_df = pd.concat([test_df, new_row_df], ignore_index=True)

    # Create a DataFrame from prediction_rows for lgbm/xgboost etc.
    future_df = pd.DataFrame(prediction_rows)
    return future_predictions, future_df
