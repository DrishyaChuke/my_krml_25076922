import pandas as pd

def generate_date_features(sales_data):
    """
    Generate date-based features such as day of the week, month, year, and weekend indicator.
    
    Parameters:
    sales_data (pd.DataFrame): Sales data with a 'date' column.
    
    Returns:
    pd.DataFrame: Sales data with additional date-based features.
    """
    # Ensure 'date' is in datetime format
    sales_data['date'] = pd.to_datetime(sales_data['date'])
    
    # Create date-based features
    sales_data['day_of_week'] = sales_data['date'].dt.dayofweek  # Monday=0, Sunday=6
    sales_data['month'] = sales_data['date'].dt.month
    sales_data['year'] = sales_data['date'].dt.year
    sales_data['is_weekend'] = sales_data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)  # 1 for weekend, 0 for weekdays
    
    return sales_data

def generate_event_features(sales_data):
    """
    Generate event-related features from 'event_name' and 'event_type' columns.
    
    Parameters:
    sales_data (pd.DataFrame): Sales data with 'event_name' and 'event_type' columns.
    
    Returns:
    pd.DataFrame: Sales data with additional event-related features.
    """
    # Convert event_name and event_type into binary features (1 if event, 0 if none)
    sales_data['is_event'] = sales_data['event_name'].apply(lambda x: 0 if x == 'None' else 1)
    sales_data['event_type_cultural'] = sales_data['event_type'].apply(lambda x: 1 if x == 'Cultural' else 0)
    sales_data['event_type_sporting'] = sales_data['event_type'].apply(lambda x: 1 if x == 'Sporting' else 0)
    sales_data['event_type_religious'] = sales_data['event_type'].apply(lambda x: 1 if x == 'Religious' else 0)
    sales_data['event_type_national'] = sales_data['event_type'].apply(lambda x: 1 if x == 'National' else 0)
    
    return sales_data

