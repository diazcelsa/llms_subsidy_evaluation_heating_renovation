import pandas as pd

def categorize_columns(df, threshold):
    # Lists to store results
    column_ids = []
    dtypes = []
    unique_vals = []

    # Iterate over each column
    for col in df.columns:
        n_unique = df[col].nunique()

        # Store column name
        column_ids.append(col)

        # Classify the column and store unique values
        if n_unique > threshold:
            dtypes.append('continuous')
            unique_vals.append(None)  # No unique values for continuous variables
        else:
            dtypes.append('categorical')
            unique_vals.append(';'.join(map(str, df[col].unique())))

    # Create a DataFrame with the results
    result_df = pd.DataFrame({
        'column_id': column_ids,
        'dtype': dtypes,
        'unique_values': unique_vals
    })

    return result_df
