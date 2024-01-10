import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



class DataFrameAnalyzer:
    def __init__(self, df, aggregation_column='year'):
        self.df = df.copy()
        self.aggregation_column = aggregation_column
        self.data_types = self._correct_data_types()
        self.potential_ordinals = self._identify_potential_ordinal_columns()
        self.recurrent_ids = self.plot_grouped_count([self.aggregation_column, 'key'])
        self.outliers = self._detect_outliers()

    def _detect_outliers(self, threshold=2):
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        outlier_df = (self.df[numeric_cols] - self.df[numeric_cols].mean()).abs() > threshold * self.df[numeric_cols].std()
        return outlier_df

    def plot_ids_per_aggregation_column(self):
        # Calculate the number of unique IDs per aggregation column
        unique_ids = self.df.groupby(self.aggregation_column)['key'].nunique()
        
        # Calculate the number of duplicate IDs per aggregation column
        duplicate_ids = self.df[self.df.duplicated(subset=['key'], keep=False)].groupby(self.aggregation_column)['key'].nunique()
        
        fig, ax = plt.subplots(figsize=(10, 5))

        # Plot the number of unique IDs using the ax object
        sns.barplot(y=unique_ids.index, x=unique_ids.values, color='blue', label='Unique IDs', orient='h', ax=ax)
        for index, value in enumerate(unique_ids):
            ax.text(value, index, str(value), color='black', ha="left", va="center")

        # Plot the number of duplicate IDs on the same ax object
        sns.barplot(y=duplicate_ids.index, x=duplicate_ids.values, color='red', label='Duplicate IDs', orient='h', ax=ax)
        for index, value in enumerate(duplicate_ids):
            ax.text(value, index, str(value), color='black', ha="left", va="center")

        ax.set_xlabel('Number of IDs')
        ax.set_ylabel(self.aggregation_column.capitalize())
        ax.set_title(f'Number of Unique and Duplicate IDs per {self.aggregation_column.capitalize()}')
        ax.legend(loc='lower right')
        plt.show()

    def _identify_potential_ordinal_columns(self, threshold=10):
        # Ensure that the data_types dictionary contains the keys 'float64' and 'float32'
        potential_columns = self.data_types.get('float64', []) + self.data_types.get('float32', [])

        potential_ordinals = []
        for col in potential_columns:
            n_unique = self.df[col].nunique()
            if 2 < n_unique <= threshold:
                potential_ordinals.append(col)
        return potential_ordinals

    def _correct_data_types(self):
        for col in self.df.columns:
            try:
                self.df[col] = pd.to_numeric(self.df[col], errors='ignore')
            except Exception as e:
                print(f"Error converting column {col}: {e}")
        
        dtype_cols = {
            'float64': [],
            'float32': [],
            'int64': [],
            'int32': [],
            'object': [],
            'category': [],
            # Include any other data types you expect
        }

        for col, dtype in self.df.dtypes.items():
            dtype_str = str(dtype)
            if dtype_str in dtype_cols:
                dtype_cols[dtype_str].append(col)
            else:
                print(f"Unexpected dtype {dtype_str} for column {col}")

        return dtype_cols
    
    def convert_float_columns_to_int(self, columns):
        """
        Convert specified columns of a DataFrame from float to integer.
        
        Parameters:
        - df: pandas DataFrame
        - columns: List of columns to convert
        
        Returns:
        - DataFrame with specified columns converted to integer
        """
        for col in columns:
            try:
                self.df[col] = self.df[col].astype(int)
            except:
                print(col)
        return self.df
    
    def plot_grouped_count(self, attributes):
        # Group by the provided attributes and count the number of occurrences
        grouped_counts = self.df.groupby(attributes).size()

        # Count the frequency of each occurrence
        frequency = grouped_counts.value_counts().reset_index()
        frequency.columns = ['Occurrences', 'Number of Keys']

        # Plotting
        plt.figure(figsize=(10, len(frequency) * 0.5))
        sns.barplot(x='Number of Keys', y='Occurrences', data=frequency, orient='h')
        plt.xlabel('Number of Keys')
        plt.ylabel('Occurrences')
        plt.title(f'Number of keys with given occurrences when grouped by {", ".join(attributes)}')
        plt.show()

    def plot_null_percentage(self):
        null_percent = (self.df.isnull().sum() / len(self.df)) * 100
        plt.figure(figsize=(10, len(null_percent) / 2))
        plot = sns.barplot(x=null_percent.values, y=null_percent.index, orient="h")
        for index, value in enumerate(null_percent):
            plot.text(value, index, f'{value:.2f}% ({self.df.isnull().sum()[index]} nulls)', color='black', ha="left", va="center", fontsize=10)
        plt.xlabel('Percentage of Nulls')
        plt.ylabel('Columns')
        plt.title('Percentage of Null Values by Column')
        plt.show()

    def plot_violin(self, columns=None):
        if columns is None:
            columns = self.df.columns.drop(['key'])

        # Calculate the number of rows required based on the number of columns
        n_rows = int(np.ceil(len(columns) / 2))

        # Create the subplots
        fig, axes = plt.subplots(n_rows, 2, figsize=(10, 5 * n_rows))
        
        # If only one row, ensure axes is a 2D array for consistent indexing
        if n_rows == 1:
            axes = np.expand_dims(axes, axis=0)

        # Plot the violin plots
        for idx, col in enumerate(columns):
            row = idx // 2
            col_idx = idx % 2

            # Check if the column has any non-NaN values
            if self.df[col].notna().any():
                sns.violinplot(x=self.df[col], ax=axes[row, col_idx])
                axes[row, col_idx].set_title(col)
            else:
                axes[row, col_idx].axis('off')
                axes[row, col_idx].set_title(f"{col} (No valid data)")

        # If there's an odd number of plots, remove the last empty subplot
        if len(columns) % 2 != 0:
            axes[-1, -1].axis('off')

        plt.tight_layout()
        plt.show()

    def plot_correlation_heatmap(self, continuous_columns):
        # Compute the correlation matrix
        corr_matrix = self.df[continuous_columns].corr()

        # Plot the heatmap
        plt.figure(figsize=(15, 15))
        sns.heatmap(corr_matrix, annot=False, fmt=".2f", cmap='coolwarm', cbar=True, square=True)

        plt.title('Correlation Matrix Heatmap')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    def plot_heatmap(self, columns=None):
        if columns is None:
            columns = self.df.columns.drop(['key'])

        # Ensure there are columns to plot
        if len(columns) == 0:
            print("No columns to plot.")
            return

        # Set the number of columns for the grid of plots
        grid_cols = 2
        grid_rows = max(1, int(np.ceil(len(columns) / grid_cols)))  # Ensure at least 1 row

        # Adjust figure size based on the number of rows
        fig_height = max(6, 2 * grid_rows)  # Reduced height multiplier to avoid too large images
        plt.figure(figsize=(20, fig_height))

        for idx, col in enumerate(columns, start=1):
            try:
                plt.subplot(grid_rows, grid_cols, idx)
                crosstab = pd.crosstab(self.df[self.aggregation_column], self.df[col])

                # Reduce the size of the heatmap if too many categories
                if crosstab.shape[0] > 50:  # Arbitrary threshold
                    print(f"Too many categories in {col} to plot.")
                    continue

                sns.heatmap(crosstab, cmap="YlGnBu", annot=True, cbar=True, annot_kws={"size": 6})
                plt.title(col)
                plt.xticks(rotation=90)
            except Exception as e:
                print(f"Error with column {col}: {e}")

        plt.tight_layout()
        plt.show()



