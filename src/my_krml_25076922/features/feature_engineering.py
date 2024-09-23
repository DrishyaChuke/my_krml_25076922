import pandas as pd

def generate_date_features(df):
    """
    Generate date-related features such as day of the week, month, and year.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing a 'date' column.
    
    Returns:
    pd.DataFrame: DataFrame with new date-related features.
    """
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek  # Monday=0, Sunday=6
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['is_weekend'] = df['day_of_week'] >= 5  # 5=Saturday, 6=Sunday
    return df

def merge_with_item_prices(sales_data, item_prices):
    """
    Merge sales data with item price data based on store_id, item_id, and week.
    
    Parameters:
    sales_data (pd.DataFrame): The reshaped sales data with date features.
    item_prices (pd.DataFrame): The item price data.
    
    Returns:
    pd.DataFrame: Sales data merged with item prices.
    """
    # Ensure date is converted to datetime and week is extracted for merging
    sales_data['wm_yr_wk'] = sales_data['date'].dt.isocalendar().week
    merged_df = sales_data.merge(item_prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
    return merged_df