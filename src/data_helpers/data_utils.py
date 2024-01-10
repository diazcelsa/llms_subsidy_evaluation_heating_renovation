import pandas as pd
import numpy as np

def transform_ordinal(df, columns, mapper, type='median'):
    """
    Transforms ordinal string values to numerical values in the specified columns 
    and replaces -2 values with the column's median.
    
    Parameters:
    - df (pd.DataFrame): The original dataframe.
    - columns (list): List of column names which are ordinal.
    - mapper (dict): Mapping dictionary with ordinal column names as keys. 
                     Each key maps to another dictionary that defines the transformation of ordinal string values to numbers.

    Returns:
    - pd.DataFrame: DataFrame after transformations.
    """
    # Step 1: replace np.nan by 'nan'
    # Step 2: Replace string ordinal values with their corresponding numerical values
    for col in columns:
        df[col] = df[col].fillna('nan')
        df[col] = df[col].map(mapper[col])
    
    # Step 3: Replace -2 values with the column's median
    for col in columns:
        if type=='median':
            subst_val = df[df[col] != -2][col].median()
        if type=='mean':
            subst_val = df[df[col] != -2][col].mean()
        df[col] = df[col].replace(-2, subst_val)

    try:
        columns_with_nulls = [col for col in columns if df[col].isna().any()]
        assert columns_with_nulls == []
    except:
        f"Columns with NaN values: {columns_with_nulls}"

    return df

def identify_strings_in_float_column(series):
    def is_float(value):
        try:
            float(value)
        except ValueError:
            return value
        
    non_float = series.apply(lambda x: is_float(x))
    return [i for i in non_float.unique() if i is not None]

def impute_missing_values(df, columns, impute_strategy='mean'):
    """
    Imputes missing values in the specified columns in the DataFrame.
    
    Parameters:
    - df (pd.DataFrame): The original dataframe.
    - columns (list): List of column names where missing values should be imputed.
    - impute_strategy (str): Strategy for imputing missing values. Can be 'mean' or 'median'. Default is 'mean'.

    Returns:
    - pd.DataFrame: DataFrame with imputed values for the specified columns.
    """
    
    if impute_strategy == 'mean':
        df[columns] = df[columns].fillna(df[columns].mean())
    elif impute_strategy == 'median':
        df[columns] = df[columns].fillna(df[columns].median())
    else:
        raise ValueError("Invalid impute_strategy. Choose either 'mean' or 'median'.")
    
    return df

def standardize_columns(df, columns_to_standardize):
    """
    Standardizes the specified columns in the DataFrame.
    
    Parameters:
    - df (pd.DataFrame): The original dataframe.
    - columns_to_standardize (list): List of column names to be standardized.

    Returns:
    - pd.DataFrame: DataFrame with standardized values for the specified columns.
    """
    
    df[columns_to_standardize] = (df[columns_to_standardize] - df[columns_to_standardize].mean()) / df[columns_to_standardize].std()
    
    return df

def dummify_categorical_columns(df, columns_to_dummify, nan_category="missing"):
    """
    Dummifies specified boolean columns in the DataFrame. Handles columns with 0/1 values,
    columns with 0/1/np.nan values, and columns with an additional category.
    
    Parameters:
    - df (pd.DataFrame): The original dataframe.
    - columns_to_dummify (list): List of column names to be dummified.
    - nan_category (str): Replacement category for np.nan values. Default is 'missing'.

    Returns:
    - pd.DataFrame: DataFrame with dummified columns.
    - dict: Dictionary mapping original column names to their corresponding dummified column names.
    """
    
    dummified_columns_dict = {}
    
    for col in columns_to_dummify:

        # Replace np.nan values with the specified category
        if np.nan in df[col].unique():
            df[col].fillna(nan_category, inplace=True)
        
        # Dummify the column
        dummies = pd.get_dummies(df[col], prefix=col)
        
        # Add to the dictionary
        dummified_columns_dict[col] = dummies.columns.tolist()
        
        # Convert the resulting dummy columns to integer type
        dummies = dummies.astype(int)
        
        # Concatenate the dummies to the original dataframe and drop the original column
        df = pd.concat([df, dummies], axis=1)
        df.drop(col, axis=1, inplace=True)
    
    return df, dummified_columns_dict

def add_prefix_to_columns(df, prefix):
    """
    Adds a prefix to specified column names in a DataFrame.

    Parameters:
    - df (pd.DataFrame): The dataframe with columns to be renamed.
    - columns (list): List of column names to be updated.
    - prefix (str): The prefix to be added to the column names.

    Returns:
    - pd.DataFrame: DataFrame with updated column names.
    """
    columns = df.columns
    rename_dict = {col: prefix + str(col) for col in columns}
    df.rename(columns=rename_dict, inplace=True)
    
    return df

def filter_subjects_by_repetitions(data, subject_col='key', within_col='year', min_repetitions=None):
    """
    Filters out subjects that have fewer repeated measurements than the specified threshold.

    Parameters:
    - data (pd.DataFrame): The dataframe containing the measurements.
    - subject_col (str): The column name for the subject/individual identifier.
    - within_col (str): The column name for the repeated measure (e.g., time, year).
    - min_repetitions (int): Minimum number of repeated measurements a subject must have to be retained.

    Returns:
    - pd.DataFrame: DataFrame with subjects having repeated measurements greater than or equal to the threshold.
    """
    
    # Count the number of repeated measurements for each subject
    subject_counts = data.groupby(subject_col)[within_col].count()
    
    # Identify subjects that meet the threshold
    valid_subjects = subject_counts[subject_counts >= min_repetitions].index
    
    # Filter the data to include only valid subjects
    filtered_data = data[data[subject_col].isin(valid_subjects)]
    
    return filtered_data

def random_sample_by_group(data, group_col='key'):
    """
    Groups data by a specified column and randomly samples one row from each group.

    Parameters:
    - data (pd.DataFrame): The dataframe containing the data.
    - group_col (str): The column name used for grouping.

    Returns:
    - pd.DataFrame: DataFrame with randomly sampled rows for each group.
    """
    
    # Group by the specified column and sample one row from each group
    sampled_data = data.groupby(group_col).apply(lambda x: x.sample(1)).reset_index(drop=True)
    
    return sampled_data