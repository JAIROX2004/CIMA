# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from tsfresh import extract_features
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
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import random


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def extract_time_series_features(df, time_column='transactionTime', id_column='atmId', value_column='balance_change'):
    """
    Extract statistical and temporal features from the time series data, handling NaN values.

    Parameters:
    - df: DataFrame with ATM time series data.
    - time_column: The name of the column representing the time dimension.
    - id_column: The column with ATM IDs.
    - value_column: The time series values (e.g., 'balance_change').
    
    Returns:
    - Extracted features DataFrame.
    """
    # Ensure the 'transactionTime' column is in the correct datetime format
    df[time_column] = pd.to_datetime(df[time_column])

    # Sort the data by ATM ID and transaction time
    df = df.sort_values(by=[id_column, time_column])

    # Handle missing values in balance_change (replace with 0, or you can choose other imputation strategies)
    df[value_column].fillna(0, inplace=True)

    # Use the tsfresh extract_features function
    extracted_features = extract_features(df[[id_column, time_column, value_column]], 
                                          column_id=id_column, 
                                          column_sort=time_column,
                                          column_value=value_column,
                                          default_fc_parameters=ComprehensiveFCParameters())

    # Handle missing values in the extracted features (replace with 0 or interpolate)
    extracted_features.fillna(0, inplace=True)

    return extracted_features


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def perform_feature_based_clustering(extracted_features, n_clusters=5):
    """
    Perform clustering using extracted time series features.

    Parameters:
    - extracted_features: DataFrame with extracted features for each ATM.
    - n_clusters: Number of clusters.

    Returns:
    - DataFrame with ATM IDs and assigned cluster labels.
    """
    # Normalize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(extracted_features)
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features_scaled)

    # Return ATM IDs and cluster labels
    atm_cluster_map = pd.DataFrame({'atmId': extracted_features.index, 'cluster_label': labels})
    
    return atm_cluster_map


def plot_clustered_time_series(df, n_clusters=5):
    """
    Plot the aggregated balance change of all ATMs for each cluster, aggregated hourly.
    Subplots will be arranged based on the number of clusters.

    Parameters:
    - df: DataFrame containing ATM data with cluster labels.
    - n_clusters: Number of clusters.
    """
    # Determine the grid size (always even)
    n_rows = n_clusters // 2
    n_cols = 2

    # Create subplots with dynamic rows/cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 4))

    # Flatten the axes for easy iteration in case we have more than 1 row
    axes = axes.flatten()

    for cluster in range(n_clusters):
        cluster_data = df[df['cluster_label'] == cluster]
        
        # Convert transactionTime to datetime and format i
        # Group by formatted transactionTime and aggregate balance_change
        aggregated_balance_change = cluster_data.groupby('transactionTime')['balance_change'].mean()

        # Plot on the appropriate axis
        ax = axes[cluster]
        ax.plot(aggregated_balance_change.index, aggregated_balance_change.values, label=f'Cluster {cluster+1}')
        ax.set_title(f"Cluster {cluster+1}: Aggregated Balance Change (Hourly)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Aggregated Balance Change")
        ax.grid(True)
        ax.tick_params(axis='x', rotation=45)
    
    # Adjust the layout to avoid overlap
    plt.tight_layout()
    plt.show()


def determine_optimal_clusters_elbow(features, max_k=10):
    """
    Determine the optimal number of clusters using the Elbow Method.

    Parameters:
    - features: Scaled feature matrix (e.g., from tsfresh).
    - max_k: Maximum number of clusters to try.

    Returns:
    - None (displays an Elbow plot).
    """
    wcss = []
    K = range(1, max_k+1)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features)
        wcss.append(kmeans.inertia_)  # Inertia is the WCSS

    # Plotting the results
    plt.figure(figsize=(8, 4))
    plt.plot(K, wcss, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.title('Elbow Method For Optimal k')
    plt.show()
from sklearn.metrics import silhouette_score

def determine_optimal_clusters_silhouette(features, max_k=10):
    """
    Determine the optimal number of clusters using Silhouette Analysis.

    Parameters:
    - features: Scaled feature matrix.
    - max_k: Maximum number of clusters to try.

    Returns:
    - None (displays a plot of average silhouette scores).
    """
    silhouette_scores = []
    K = range(2, max_k+1)  # Silhouette score is not defined for k=1
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(features)
        score = silhouette_score(features, labels)
        silhouette_scores.append(score)
        print(f'For k={k}, the average silhouette_score is : {score}')

    # Plotting the results
    plt.figure(figsize=(8, 4))
    plt.plot(K, silhouette_scores, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Average Silhouette Score')
    plt.title('Silhouette Analysis For Optimal k')
    plt.show()

from sklearn.metrics import davies_bouldin_score

def determine_optimal_clusters_davies_bouldin(features, max_k=10):
    """
    Determine the optimal number of clusters using the Davies-Bouldin Index.

    Parameters:
    - features: Scaled feature matrix.
    - max_k: Maximum number of clusters to try.

    Returns:
    - None (displays a plot of DBI scores).
    """
    dbi_scores = []
    K = range(2, max_k+1)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(features)
        score = davies_bouldin_score(features, labels)
        dbi_scores.append(score)
        print(f'For k={k}, the Davies-Bouldin Index is : {score}')

    # Plotting the results
    plt.figure(figsize=(8, 4))
    plt.plot(K, dbi_scores, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Davies-Bouldin Index')
    plt.title('Davies-Bouldin Index For Optimal k')
    plt.show()


import numpy as np
from numpy.linalg import inv

def min_trace_reconciliation(cluster_forecast, atm_forecasts, cov_matrix):
    """
    Applies the MinT reconciliation method to adjust ATM forecasts to align with the cluster forecast.
    
    Args:
        cluster_forecast (array): Forecast for the cluster level (shape: [time_steps]).
        atm_forecasts (list of arrays): List of individual ATM forecasts (shape: [n_atms, time_steps]).
        cov_matrix (ndarray): Covariance matrix of ATM forecast errors (shape: [n_atms, n_atms]).
        
    Returns:
        reconciled_atm_forecasts (array): Reconciled ATM-level forecasts (shape: [n_atms, time_steps]).
    """
    n_atms = len(atm_forecasts)
    time_steps = len(cluster_forecast)
    
    # Convert atm_forecasts to a 2D array (shape: [n_atms, time_steps])
    atm_forecasts_matrix = np.vstack(atm_forecasts)
    
    # Initialize a matrix to store reconciled forecasts (same shape as atm_forecasts_matrix)
    reconciled_atm_forecasts = np.zeros_like(atm_forecasts_matrix)
    
    # G matrix: a column of ones representing the structure (sum of ATMs)
    G = np.ones((n_atms, 1))
    
    # Loop through each time step and reconcile forecasts
    for t in range(time_steps):
        # Forecast sum at time step t
        atm_sum_forecast_t = np.sum(atm_forecasts_matrix[:, t])
        
        # Residual at time step t (difference between cluster forecast and sum of ATM forecasts)
        residual_t = cluster_forecast[t] - atm_sum_forecast_t
        
        # Calculate S_inv (inverse of covariance matrix)
        S_inv = inv(cov_matrix)
        
        # MinT reconciliation formula: adjust_factor = (G.T @ S_inv @ G)^-1 @ G.T @ S_inv @ residual_t
        adjust_factor_t = inv(G.T @ S_inv @ G) @ (G.T @ S_inv @ residual_t)
        
        # Adjust ATM forecasts at time step t
        reconciled_atm_forecasts[:, t] = atm_forecasts_matrix[:, t] + G @ adjust_factor_t
    
    return reconciled_atm_forecasts
# Example usage:
'''
# Example usage:
cluster_forecast = np.array(cluster_results[24]['LightGBM']['future_predictions'])  # Cluster-level forecast
atm_forecasts = [np.array(atm_results[atm_id]['LightGBM']['future_predictions']) for atm_id in atm_results]  # ATM forecasts

# Covariance matrix based on historical forecast errors (must be of shape [n_atms, n_atms])
cov_matrix = np.cov([np.array(atm_results[atm_id]['LightGBM']['predictions'][-34:]) - 
                     np.array(atm_results[atm_id]['LightGBM']['future_predictions'][-34:])
                     for atm_id in atm_results])

# You now have reconciled ATM-level forecasts.
reconciled_atm_forecasts = min_trace_reconciliation(cluster_forecast, atm_forecasts, cov_matrix)

def min_trace_reconciliation(cluster_forecast, atm_forecasts, atm_predictions, cov_matrix):
    """
    Applies the MinT reconciliation method to adjust ATM forecasts to align with the cluster forecast.

    Args:
        cluster_forecast (array): Forecast for the cluster level.
        atm_forecasts (list): List of individual ATM forecasts (arrays).
        atm_predictions (array): Array of predicted values for each ATM.
        cov_matrix (ndarray): Covariance matrix of ATM forecast errors.

    Returns:
        reconciled_atm_forecasts (array): Reconciled ATM-level forecasts.
    """
    # Sum of ATM forecasts to compare with cluster forecast
    atm_sum_forecast = np.sum(atm_forecasts, axis=0)
    
    # Difference between the cluster forecast and the sum of ATM forecasts
    residual = cluster_forecast - atm_sum_forecast

    # Compute the weights for reconciliation (based on covariance matrix)
    G = np.ones((len(atm_forecasts), 1))  # Hierarchical structure
    H = np.concatenate(atm_forecasts, axis=0).reshape(-1, 1)

    # MinT reconciliation formula: H_adj = H + (G.T @ (S^-1) @ G)^-1 @ G.T @ (S^-1) @ residual
    S_inv = inv(cov_matrix)
    adjust_factor = inv(G.T @ S_inv @ G) @ (G.T @ S_inv @ residual)

    # Adjust ATM forecasts
    reconciled_atm_forecasts = np.array(atm_forecasts) + adjust_factor

    return reconciled_atm_forecasts

# Example usage:

# Let's say we have the aggregated cluster forecast from cluster_results for a particular model (e.g., LightGBM):
cluster_forecast = np.array(cluster_results[3]['LightGBM']['predictions'])

# And we have individual ATM forecasts (LightGBM predictions):
atm_forecasts = [np.array(atm_results[atm_id]['LightGBM']['predictions']) for atm_id in atm_results]
print(atm_forecasts)
# Assume you have a covariance matrix of the forecast errors (this could be estimated from past errors):
# Apply MinT reconciliation
reconciled_atm_forecasts = min_trace_reconciliation(cluster_forecast, atm_forecasts, atm_forecasts, cov_matrix)

# You now have reconciled ATM-level forecasts.
'''