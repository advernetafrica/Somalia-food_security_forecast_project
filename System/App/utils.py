import pandas as pd

def filter_data(df, county=None, sub_county=None, facility=None, commodities=None):
    """
    Filter the dataset based on selected location and commodities
    
    Parameters:
    df (pandas.DataFrame): The dataset to filter
    county (str): County name filter
    sub_county (str): Sub-county name filter
    facility (str): Facility name filter
    commodities (list): List of commodities to include
    
    Returns:
    pandas.DataFrame: Filtered dataframe
    """
    filtered_df = df.copy()
    
    if county:
        filtered_df = filtered_df[filtered_df["county_name"] == county]
    
    if sub_county:
        filtered_df = filtered_df[filtered_df["sub_county_name"] == sub_county]
        
    if facility:
        filtered_df = filtered_df[filtered_df["facility_name"] == facility]
        
    if commodities:
        filtered_df = filtered_df[filtered_df["dataelement_name"].isin(commodities)]
        
    return filtered_df


def calculate_lag_features(df, period_col="period", value_col="value", n_lags=12):
    """
    Calculate lag features for time series analysis
    
    Parameters:
    df (pandas.DataFrame): Time series data
    period_col (str): Column name for time periods
    value_col (str): Column name for values
    n_lags (int): Number of lag periods to calculate
    
    Returns:
    dict: Dictionary containing lag values and rolling means
    """
    if df.empty:
        return {
            "lag_1": 0,
            "lag_3": 0,
            "rolling_mean_3": 0
        }
        
    # Sort by period for correct lag calculation
    df = df.sort_values(period_col)
    
    # Get the most recent values
    recent_values = df[value_col].tail(n_lags).tolist()
    
    # Calculate lag features
    lag_1 = recent_values[-1] if len(recent_values) >= 1 else 0
    lag_3 = recent_values[-3] if len(recent_values) >= 3 else 0
    rolling_mean_3 = sum(recent_values[-3:]) / 3 if len(recent_values) >= 3 else 0
    
    return {
        "lag_1": lag_1,
        "lag_3": lag_3,
        "rolling_mean_3": rolling_mean_3
    }