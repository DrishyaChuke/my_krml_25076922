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