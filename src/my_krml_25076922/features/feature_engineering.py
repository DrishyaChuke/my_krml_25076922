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
