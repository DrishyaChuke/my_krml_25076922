import pandas as pd
import numpy as np

import gc
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


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
    Replace missing sell_price values with 0 (assuming no sales for the product on that day).
    
    Parameters:
    sales_data (pd.DataFrame): The reshaped sales data with date features.
    item_prices (pd.DataFrame): The item price data.
    
    Returns:
    pd.DataFrame: Sales data merged with item prices, with null sell_price replaced with 0.
    """
    # Merge sales data with item prices
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

def check_if_integer(series):
    """Check if a float column can be safely cast to integer."""
    return np.all(np.mod(series, 1) == 0)

def reduce_mem_usage(df, int_cast=True, obj_to_category=False, subset=None):
    """
    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.
    :param df: dataframe to reduce (pd.DataFrame)
    :param int_cast: indicate if columns should be tried to be casted to int (bool)
    :param obj_to_category: convert non-datetime related objects to category dtype (bool)
    :param subset: subset of columns to analyse (list)
    :return: dataset with the column dtypes adjusted (pd.DataFrame)
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2;
    gc.collect()
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    cols = subset if subset is not None else df.columns.tolist()

    for col in tqdm(cols):
        col_type = df[col].dtype

        if col_type != object and col_type.name != 'category' and 'datetime' not in col_type.name:
            c_min = df[col].min()
            c_max = df[col].max()

            # test if column can be converted to an integer
            treat_as_int = str(col_type)[:3] == 'int'
            if int_cast and not treat_as_int:
                treat_as_int = check_if_integer(df[col])

            if treat_as_int:
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.uint8).min and c_max < np.iinfo(np.uint8).max:
                    df[col] = df[col].astype(np.uint8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.uint16).min and c_max < np.iinfo(np.uint16).max:
                    df[col] = df[col].astype(np.uint16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.uint32).min and c_max < np.iinfo(np.uint32).max:
                    df[col] = df[col].astype(np.uint32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
                elif c_min > np.iinfo(np.uint64).min and c_max < np.iinfo(np.uint64).max:
                    df[col] = df[col].astype(np.uint64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        elif 'datetime' not in col_type.name and obj_to_category:
            df[col] = df[col].astype('category')
    gc.collect()
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.3f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

