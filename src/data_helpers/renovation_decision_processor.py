import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm
from sklearn.neighbors import NearestNeighbors
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor


def calculate_efficiency_renovation_metrics(df, target_column, filter_out=0, group='a'):
    df_ = df.copy()
 
    # Energy efficiency
    df_['ener_save_simple_kwh'] = df_['ebj'] - df_['ebe']
    df_['ener_save_complex_kwh'] = df_['ebj'] - df_['ebu']
    df_['ener_save_simple'] = df_['nist5'] * df_['ener_save_simple_kwh']
    df_['ener_save_complex'] = df_['nist5'] * df_['ener_save_complex_kwh']
    df_['ener_save_gain_complex_kw'] = df_['ener_save_complex'] - df_['ener_save_simple']

    # Cost efficiency
    df_['cost_save_simple_eur'] = df_['kdj'] - df_['kde']
    df_['cost_save_complex_eur'] = df_['kdj'] - df_['kdu']
    df_['cost_save_gain_complex_eur'] = df_['cost_save_complex_eur'] - df_['cost_save_simple_eur']
    df_['cost_save_gain_complex_10_y_eur'] = df_['kf10']
    df_['cost_save_gain_complex_10_y_+2_eur'] = df_['kf2st']
    df_['cost_save_gain_complex_10_y_-2_eur'] = df_['kf2si']
 
    # Normalize the target
    if filter_out != 0:
        df_ = df_[df_[target_column]!=filter_out]

    if group == 'a':
        mapper = {1:0, 2:1}
    elif group == 'b':
        mapper = {1:1, 2:0}
    else:
        raise ValueError
    df_[target_column] = df_[target_column].map(mapper)

    return df_

def process_regional_information(df):
    # Define the IDs for West Germany
    east_ids = [11, 12, 13, 14, 15, 16]  # Example IDs for East Germany

    # Define IDs for Berlin and Brandenburg
    berlin_brandenburg_ids = [11, 12]  # Assuming 3 is Berlin and 11 is Brandenburg

    # Create boolean columns
    df['is_east'] = df['bundesland'].isin(east_ids)
    df['is_ber_brand'] = df['bundesland'].isin(berlin_brandenburg_ids)

    #     Small City: Less than 20,000 inhabitants
    #     Mid-size City: 20,000 to 100,000 inhabitants
    #     Large City: More than 100,000 inhabitants
    df_ = df.copy()
    region_mapper = pd.read_csv("/Users/celsa/github/thesis/data/ariadne/heating_buildings/gemainde_mapper.csv")
    region_mapper = region_mapper[['plz','city_category']].drop_duplicates(subset=['plz'])
    df_ = df_.merge(region_mapper, on='plz', how='left')
    print(df.shape, df_.shape)

    assert len(df) == len(df_)

    # Find PLZ in df that are not in the region mapper
    missing_plz = df_[df_['city_category'].isnull()]['plz'].unique()

    # Initialize Nearest Neighbors model
    nn = NearestNeighbors(n_neighbors=1)

    # Fit model on available PLZ in region mapper
    nn.fit(region_mapper[['plz']])

    # Function to find nearest PLZ
    def find_nearest_plz(plz):
        distance, index = nn.kneighbors([[plz]])
        return region_mapper.iloc[index[0][0]]['plz']

    # Impute missing city categories
    for plz in missing_plz:
        nearest_plz = find_nearest_plz(plz)
        category = region_mapper[region_mapper['plz'] == nearest_plz]['city_category'].values[0]
        df_.loc[df_['plz'] == plz, 'city_category'] = category

    df_['city_category'] = df_['city_category'].map({'small':1,'mid-size':2,'large':3})

    assert df.isnull().sum().sum() == df_.isnull().sum().sum()
    return df_
    
         
def process_rename_columns_bool(df, boolean_columns_mapping, mapping, rename_columns_dict):
    """
    Processes a DataFrame by mapping boolean column values and renaming columns.

    Parameters:
    - df: A pandas DataFrame to be processed.
    - boolean_columns_mapping: A dictionary with column names as keys and mapping dictionaries as values.
      Example: {'column1': {True: 'Yes', False: 'No'}, 'column2': {True: 1, False: 0}}
    - rename_columns_dict: A dictionary for renaming columns. Keys are original column names, and values are new names.
      Example: {'old_name1': 'new_name1', 'old_name2': 'new_name2'}

    Returns:
    - The processed DataFrame.
    """
    # Map values in boolean columns
    for column in boolean_columns_mapping:
        if column in df.columns:
            df[column] = df[column].map(mapping)

    # Rename columns
    df.rename(columns=rename_columns_dict, inplace=True)

    return df

def is_boolean_column(series):
    """
    Checks if a Pandas Series is a boolean column, containing only 0/1 or True/False.
    """
    # Check if the input is a pandas Series
    if not isinstance(series, pd.Series):
        raise TypeError(f"Expected input to be a pandas Series, but got {type(series)} instead.")

    # Drop NaN values and get unique values
    unique_values = series.dropna().unique()

    # Check if the unique values are either {0, 1} or {False, True}
    return set(unique_values).issubset({0, 1, False, True})

def preprocess_and_fit_regression(df, dummify_cols, target_col, regression_type='ols', binary_mapping=None, columns_to_drop=[]):
    """
    Preprocess the DataFrame and fit a regression model (OLS or Logistic).

    Parameters:
    df (pd.DataFrame): The DataFrame to preprocess.
    dummify_cols (list): List of columns to dummify.
    target_col (str): The name of the target variable column.
    regression_type (str): Type of regression ('ols' for OLS regression, 'logit' for Logistic regression).
    binary_mapping (dict, optional): Mapping for binary logistic regression target variable.

    Returns:
    Regression model results.
    """
    # Dummify categorical columns
    df_ = pd.get_dummies(df, columns=dummify_cols, drop_first=True).reset_index(drop=True)

    # Drop specified dummified columns if provided
    if columns_to_drop:
        df_.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    # Identify boolean columns
    boolean_cols = []
    for col in df_.columns:
        print(f"Column {col} data type:", type(df_[col]))  # This will print the type of df_[col]
        if col != target_col:
            try:
                # Now check if df_[col] is a Series and contains boolean values
                if is_boolean_column(df_[col]):
                    boolean_cols.append(col)
            except TypeError as e:
                print(f"Error with column {col}: {e}")
                # Optionally, you can add a breakpoint or continue to the next iteration
                continue 

    # Identify boolean columns, excluding the target column
    boolean_cols = [col for col in df_.columns if col != target_col and is_boolean_column(df_[col])]

    # Fill null values in boolean columns with their mode
    for col in boolean_cols:
        mode_value = df_[col].mode()[0]
        df_[col] = df_[col].fillna(mode_value).astype(int)

    # Exclude columns to be dummified or are boolean from standardization
    standardize_cols = [col for col in df_.columns if col not in dummify_cols + boolean_cols and col != target_col]

    # Preprocessing pipelines
    continuous_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Combine preprocessing for continuous columns
    preprocessor = ColumnTransformer([
        ('num', continuous_pipeline, standardize_cols)
    ])

    # Fit and transform the data
    X_transformed = preprocessor.fit_transform(df_[standardize_cols])
    X_processed = pd.DataFrame(X_transformed, columns=standardize_cols).reset_index(drop=True)

    # Add dummified and boolean columns back to X_processed
    X_processed = pd.concat([X_processed, df_[boolean_cols]], axis=1)
    #print(X_processed.columns, print(X_processed['group'].unique()), print(X_processed['order'].unique()))

    non_numeric_cols = X_processed.select_dtypes(include=['object']).columns
    if len(non_numeric_cols) > 0:
        print("Non-numeric columns found:", non_numeric_cols)
    print(X_processed.dtypes.tolist())
    print(f"# nulls: {X_processed.isnull().sum().sum()}")
    print(df_[target_col].unique())

    # Reset index before fitting the model
    X_processed.reset_index(drop=True, inplace=True)
    y = df_[target_col].reset_index(drop=True)

    # Plotting the correlation matrix heatmap
    if binary_mapping:
        corr_matrix = pd.concat([X_processed, y.map(binary_mapping)], axis=1).corr()
    else:
        corr_matrix = pd.concat([X_processed], axis=1).corr()
    plt.figure(figsize=(15, 15))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
    plt.title('Correlation Matrix Heatmap')
    plt.show()

        # Convert target variable for logistic regression
    if regression_type == 'logit':
        if binary_mapping:
            y = y.map(binary_mapping)
            # Check if y contains only 0 and 1
            if not np.all(np.isin(y.unique(), [0, 1])):
                raise ValueError("Target variable after mapping must contain only 0 and 1.")
        else:
            raise ValueError("Binary mapping for target variable is required for logistic regression.")

    # Identify and print non-numeric columns
    non_numeric_columns = X_processed.select_dtypes(include=['object']).columns
    if len(non_numeric_columns) > 0:
        print("Object-type columns in processed DataFrame:", non_numeric_columns)

    # Convert DataFrame to numpy arrays
    X_processed_np = X_processed.to_numpy()
    y_np = df_[target_col].to_numpy()

    # List of feature names, including the constant term
    feature_names = ['const'] + X_processed.columns.tolist()

    # Add a constant to the model (for the intercept)
    X_with_const = sm.add_constant(X_processed_np)

    # Fit the model based on the specified regression type
    if regression_type == 'ols':
        model = sm.OLS(y_np, X_with_const).fit()
    elif regression_type == 'logit':
        model = sm.Logit(y_np, X_with_const).fit()
    else:
        raise ValueError("Invalid regression type specified. Choose 'ols' or 'logit'.")

    # Print model summary with feature names
    print(model.summary(xname=feature_names))

    return model, feature_names, X_processed
