import pandas as pd

from sklearn.preprocessing import StandardScaler


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
    merged_df = sales_data.merge(calendar_data[['d', 'date', 'wm_yr_wk']], on='d', how='left')
    
    return merged_df

def merge_with_item_prices(sales_data, item_prices):
    """
    Merge sales data with item price data based on store_id, item_id, and wm_yr_wk.
    
    Parameters:
    sales_data (pd.DataFrame): Sales data that already contains the wm_yr_wk column.
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

def impute_missing_prices(sales_data):
    """
    Impute missing sell_price values using forward fill, backward fill, and median imputation.
    
    Parameters:
    sales_data (pd.DataFrame): Sales data with missing sell_price values.
    
    Returns:
    pd.DataFrame: Sales data with missing prices imputed.
    """
    # Forward fill missing prices based on item_id
    sales_data['sell_price'] = sales_data.groupby('item_id')['sell_price'].fillna(method='ffill')
    
    # Backward fill any remaining missing prices
    sales_data['sell_price'] = sales_data.groupby('item_id')['sell_price'].fillna(method='bfill')
    
    # Median imputation for any remaining missing prices
    sales_data['sell_price'] = sales_data.groupby('item_id')['sell_price'].transform(lambda x: x.fillna(x.median()))
    
    return sales_data

# Add this in my_krml_25076922/features/data_preprocessing.py

from sklearn.preprocessing import LabelEncoder

def encode_categorical_columns(sales_data, categorical_columns):
    """
    Apply label encoding to categorical columns in the sales data.
    
    Parameters:
    sales_data (pd.DataFrame): Sales data containing categorical features.
    categorical_columns (list): List of categorical columns to encode.
    
    Returns:
    pd.DataFrame: Sales data with encoded categorical features.
    """
    le = LabelEncoder()
    for col in categorical_columns:
        sales_data[col] = le.fit_transform(sales_data[col].astype(str))
    
    return sales_data

from sklearn.model_selection import train_test_split

def split_data(sales_data, target_column, test_size=0.2, random_state=42):
    """
    Split the sales data into training and validation sets.
    
    Parameters:
    sales_data (pd.DataFrame): Sales data containing features and target column.
    target_column (str): The column name of the target variable (sales).
    test_size (float): The proportion of the dataset to include in the validation set.
    random_state (int): Random state for reproducibility.
    
    Returns:
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series: Training features, validation features, training target, validation target.
    """
    X = sales_data.drop(columns=[target_column])
    y = sales_data[target_column]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_val, y_train, y_val

def scale_numerical_columns(sales_data, numerical_columns):
    """
    Apply standard scaling to numerical columns in the sales data.
    
    Parameters:
    sales_data (pd.DataFrame): Sales data containing numerical features.
    numerical_columns (list): List of numerical columns to scale.
    
    Returns:
    pd.DataFrame: Sales data with scaled numerical features.
    """
    scaler = StandardScaler()
    sales_data[numerical_columns] = scaler.fit_transform(sales_data[numerical_columns])
    
    return sales_data