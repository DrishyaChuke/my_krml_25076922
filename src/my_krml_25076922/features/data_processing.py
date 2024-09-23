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
    Merge sales data with calendar data to replace 'd' values with actual dates.
    
    Parameters:
    sales_data (pd.DataFrame): The reshaped sales data in long format (with 'd' column).
    calendar_data (pd.DataFrame): The calendar data containing 'd' and 'date' columns.
    
    Returns:
    pd.DataFrame: Sales data with actual dates from the calendar.
    """
    # Merge sales data with calendar to get actual dates
    merged_df = sales_data.merge(calendar_data[['d', 'date']], on='d', how='left')
    
    return merged_df

def merge_with_item_prices(sales_data, item_prices):
    """
    Merge sales data with item price data based on store_id, item_id, and wm_yr_wk.
    
    Parameters:
    sales_data (pd.DataFrame): Sales data with date features (and calculated week of the year).
    item_prices (pd.DataFrame): Item price data containing store_id, item_id, and wm_yr_wk.
    
    Returns:
    pd.DataFrame: Sales data merged with item prices.
    """
    
    merged_df = sales_data.merge(item_prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
    
    return merged_df

def merge_with_events(sales_data, events_data):
    """
    Merge sales data with events data to add event-related features.
    
    Parameters:
    sales_data (pd.DataFrame): Sales data with date features.
    events_data (pd.DataFrame): Events data with 'date', 'event_name', and 'event_type' columns.
    
    Returns:
    pd.DataFrame: Sales data with event-related features.
    """
    # Merge sales data with event information
    merged_df = sales_data.merge(events_data[['date', 'event_name', 'event_type']], on='date', how='left')
    
    # Fill missing events with 'None'
    merged_df['event_name'] = merged_df['event_name'].fillna('None')
    merged_df['event_type'] = merged_df['event_type'].fillna('None')
    
    return merged_df
