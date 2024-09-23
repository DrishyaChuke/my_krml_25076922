import pandas as pd

def reshape_sales_data(df, id_vars):
    """
    Reshape sales data from wide format to long format using pd.melt().
    
    Parameters:
    df (pd.DataFrame): The sales dataframe in wide format.
    id_vars (list): List of columns to keep (id columns like 'item_id', 'store_id', etc.)
    
    Returns:
    pd.DataFrame: Reshaped sales dataframe in long format.
    """
    reshaped_df = pd.melt(
        df,
        id_vars=id_vars,  # Columns to keep as-is
        var_name='d',      # Name for the melted 'day' column
        value_name='sales' # Name for the melted 'sales' column
    )
    return reshaped_df

def merge_with_calendar(sales_data, calendar_data):
    """
    Merge reshaped sales data with calendar to get actual dates.
    
    Parameters:
    sales_data (pd.DataFrame): The reshaped sales dataframe.
    calendar_data (pd.DataFrame): The calendar dataframe with 'd' and 'date' columns.
    
    Returns:
    pd.DataFrame: Sales data with actual dates.
    """
    merged_df = sales_data.merge(calendar_data[['d', 'date']], on='d', how='left')
    return merged_df

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