import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.utils import resample

import seaborn as sns
import matplotlib.pyplot as plt

class ResultEvaluator:
    def __init__(self, final_columns, opportunity_costs):
        self.final_columns = final_columns
        self.opportunity_cost = opportunity_costs

    def transform_options_to_probabilities(self, df):
        df_ = df.copy()
        for col in self.final_columns:
            df_[col] = df_[col].map({1: 0, 2: 1})
        return df_

    def format_initial_data(self, df, key_ids, econometric_factors, optional_factors):
        """
        Perform initial data formatting specific to the dataset.
        """
        a4 = {1: 'freistehendes Ein-/ Zweifamilienhaus', 2: 'Reihen-/Doppelhaus', 
              3: 'in einem Mehrfamilienhaus (bis zu sieben Stockwerke)'}
        df = df.rename(columns={'a4': "building_type", "a1": "n_housholds"})
        df['n_housholds'] = df['n_housholds'].replace(-1, np.nan)
        df['n_housholds'] = df['n_housholds'].fillna(df['n_housholds'].mean())
        df['building_type'] = df['building_type'].map(a4)
        df_ = df[(df['citizen_type']!='unknown') & (df['key'].isin(key_ids))]
        df_ = df_[['key']+econometric_factors+optional_factors+self.final_columns+['citizen_type']]
        df_ = self.transform_options_to_probabilities(df_)
        return df_

    def pivot_with_opportunity_cost(self, df):
        pivoted_df = pd.DataFrame()

        for index, row in df.iterrows():
            for i, col in enumerate(self.final_columns):
                # Create a temporary DataFrame for each row's transformation
                temp_df = pd.DataFrame({
                    'key': [row['key']],
                    'decision': [col],
                    'choice': [row[col]],
                    'opportunity_cost': [self.opportunity_cost[i]]
                })
                try:
                    temp_df['label'] = row['label']
                    temp_df['temperature'] = row['temperature']
                    temp_df['iter'] = row['iter']
                    temp_df['success'] = row['success']
                except:
                    temp_df['label'] = 'human'
                    temp_df['group'] = row['citizen_type']

                # Append to the main pivoted DataFrame
                pivoted_df = pd.concat([pivoted_df, temp_df], ignore_index=True)
        return pivoted_df
    
    def categorize_customer_type_row(self, row):
        if all(row[col] == 3 for col in self.final_columns):
            return 'unknown'
        elif all(row[col] == 1 for col in self.final_columns):
            return 'never taker'
        elif all(row[col] == 2 for col in self.final_columns):
            return 'free rider'
        elif any(row[col] == 1 for col in self.final_columns) and any(row[col] == 2 for col in self.final_columns):
            # from 2-5 low
            if row['ea801'] == 2 and any(row[col] == 1 for col in ['ea802', 'ea803', 'ea804', 'ea805']):
                return 'complier'#'hard_complier'
            # from 6-10 middle
            elif row['ea805'] == 2 and any(row[col] == 1 for col in ['ea806', 'ea807', 'ea808', 'ea809', 'ea810']):
                return 'complier'#'midle_complier'
            # from 11 to 15 high
            elif row['ea810'] == 2 and any(row[col] == 1 for col in ['ea811', 'ea812', 'ea813', 'ea814', 'ea815']):
                return 'complier'#'easy_complier'
            else:
                return 'defier'
        else:
            return 'other'  # For any rows that don't fit the above categories
    
    def prepare_for_ols_logistic_regression(self, df):
        # List of boolean columns
        boolean_columns = ['is_east', 'is_high_wealth', 'is_altruist']

        # Convert boolean columns to 0/1
        for col in boolean_columns:
            df[col] = df[col].astype(int)

        # List of categorical columns
        categorical_columns = ['total_income_level', 'city_size', 'high_education_level', 
                            'building_period', 'prof_status', 'pol_orientation', 
                            'nature_level', 'ownership_level', 'profit_focus',
                            'building_type']

        # Dummify categorical columns
        for col in categorical_columns:
            dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
            df = pd.concat([df, dummies], axis=1)
            df.drop(col, axis=1, inplace=True)

        return df
    
    def fit_ols_logistic_regression(self, df, dependent_var, columns_to_exclude):
        # Filter out the columns not needed for the regression
        df_filtered = df.drop(columns=columns_to_exclude + self.final_columns)

        # Separate the dependent variable
        y = df_filtered[dependent_var]

        # Independent variables
        X = df_filtered.drop(dependent_var, axis=1)

        # Add a constant to the model (intercept)
        X = sm.add_constant(X)

        # Fit the OLS logistic regression model
        model = sm.OLS(y, X).fit()

        return model, model.summary()
    
    def concatenate_and_transform_dfs(self, df_human, df_synthetic):
        # Assign new column 'is_synthetic' based on the DataFrame origin
        df_human['is_synthetic'] = 0
        df_synthetic['is_synthetic'] = 1

        # Concatenate the two DataFrames
        concatenated_df = pd.concat([df_human, df_synthetic], ignore_index=True)

        return concatenated_df
    
    def collect_metrics_across_conditions(self, df_human, synthetic_dataframes, classes, temperatures, prompt_type, econometric_factors, optional_factors, n_bootstraps):
        # Initialize a list to store metric dictionaries
        all_metrics = []
        
        # Set up the matplotlib figure for a grid of plots
        num_synthetic_dfs = len(synthetic_dataframes)
        fig, axes = plt.subplots(1, num_synthetic_dfs, figsize=(20, 5))
        
        # Flatten the axes array for easy iteration if there's more than one row
        # Correctly handle the axes indexing when there's only one row
        if num_synthetic_dfs == 1:
            axes = [axes]
        
        # Iterate over synthetic dataframes and temperatures
        for i, df_synthetic in enumerate(synthetic_dataframes):
            # prepare dataframe
            samples = df_synthetic['key'].unique().tolist()
            df_sample_sel = self.format_initial_data(df_human, samples, econometric_factors, optional_factors)
            df_synthetic['citizen_type'] = df_synthetic.apply(self.categorize_customer_type_row, axis=1)
            
            # Call the plot_confusion_matrix function
            metrics = self.calculate_confusion_matrix_with_bootstrapping(df_sample_sel, df_synthetic, classes, temperatures[i], prompt_type, n_bootstraps)
            all_metrics.append(metrics)
            
            # Plot the confusion matrix on the designated axes within the grid
            ax = axes[i]  # Correct indexing for axes
            sns.heatmap(metrics['cm'], annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
            ax.set_title(f'Temp: {temperatures[i]}, Prompt: {prompt_type}')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            
            # Add the metrics text to the right of the heatmap
            metrics_text = (f'Aggregated Precision: {metrics["precision_agg"]:.2f}\n'
                            f'Aggregated Recall: {metrics["recall_agg"]:.2f}\n'
                            f'Aggregated F1-Score: {metrics["f1_score_agg"]:.2f}\n\n'
                            f'Aggregated Precision SE: {metrics["precision_agg_se"]:.2f}\n'
                            f'Aggregated Recall SE: {metrics["recall_agg_se"]:.2f}\n'
                            f'Aggregated F1-Score SE: {metrics["f1_score_agg_se"]:.2f}\n\n'
                            'Individual Class Metrics:\n' +
                            '\n'.join([f'{cls} - P: {p:.2f}, R: {r:.2f}, F1: {f:.2f}' 
                                    for cls, p, r, f in zip(classes, metrics["precision"], metrics["recall"], metrics["f1_score"])]))
            ax.text(1.3, 0.5, metrics_text, transform=ax.transAxes, fontsize=9, verticalalignment='center')
            for ax in axes:
                ax.set_xlabel('Predicted')  # Set x-axis label
                ax.set_ylabel('True')
        
        # Adjust the layout
        plt.tight_layout()
        plt.subplots_adjust(wspace=1)
        plt.show()
        
        # Convert the list of metric dictionaries to a dataframe
        metrics_df = pd.DataFrame(all_metrics)
        
        return metrics_df, df_sample_sel, df_synthetic
    
    def calculate_confusion_matrix_with_bootstrapping(self, df_human, df_synthetic, classes, temperature, prompt_type, n_bootstraps=1000):
        # Initial metrics calculation
        df_combined = df_human[['key', 'citizen_type']].rename(columns={'citizen_type': 'citizen_type_true'}) \
            .merge(df_synthetic[['key', 'citizen_type']].rename(columns={'citizen_type': 'citizen_type_pred'}), on='key')
        df_combined = df_combined[~(df_combined['citizen_type_true'].isin(['unknown', 'defier']))]
        y_true = df_combined['citizen_type_true']
        y_pred = df_combined['citizen_type_pred']
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        
        precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
        precision_agg, recall_agg, f1_score_agg, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)

        # Bootstrapping for standard errors
        bootstrap_precisions, bootstrap_recalls, bootstrap_f1_scores = (
            {class_: [] for class_ in classes} for _ in range(3)
        )
        bootstrap_precision_agg, bootstrap_recall_agg, bootstrap_f1_score_agg = [], [], []

        for _ in range(n_bootstraps):
            df_sampled = resample(df_combined)
            y_true_sampled = df_sampled['citizen_type_true']
            y_pred_sampled = df_sampled['citizen_type_pred']

            sample_precision, sample_recall, sample_f1_score, _ = precision_recall_fscore_support(
                y_true_sampled, y_pred_sampled, labels=classes, zero_division=0
            )
            sample_precision_agg, sample_recall_agg, sample_f1_score_agg, _ = precision_recall_fscore_support(
                y_true_sampled, y_pred_sampled, average='macro', zero_division=0
            )

            bootstrap_precision_agg.append(sample_precision_agg)
            bootstrap_recall_agg.append(sample_recall_agg)
            bootstrap_f1_score_agg.append(sample_f1_score_agg)

            for i, class_ in enumerate(classes):
                bootstrap_precisions[class_].append(sample_precision[i])
                bootstrap_recalls[class_].append(sample_recall[i])
                bootstrap_f1_scores[class_].append(sample_f1_score[i])

        precision_se = {class_: np.std(bootstrap_precisions[class_]) for class_ in classes}
        recall_se = {class_: np.std(bootstrap_recalls[class_]) for class_ in classes}
        f1_score_se = {class_: np.std(bootstrap_f1_scores[class_]) for class_ in classes}
        precision_agg_se = np.std(bootstrap_precision_agg) / np.sqrt(n_bootstraps)
        recall_agg_se = np.std(bootstrap_recall_agg) / np.sqrt(n_bootstraps)
        f1_score_agg_se = np.std(bootstrap_f1_score_agg) / np.sqrt(n_bootstraps)

        # Metrics dictionary
        metrics = {
            'temperature': temperature,
            'prompt_type': prompt_type,
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1_score': f1_score.tolist(),
            'precision_agg': precision_agg,
            'recall_agg': recall_agg,
            'f1_score_agg': f1_score_agg,
            'precision_agg_se': precision_agg_se,
            'recall_agg_se': recall_agg_se,
            'f1_score_agg_se': f1_score_agg_se,
            'cm': cm.tolist(),
            'n_samples': len(y_true),
            'precision_se': precision_se,
            'recall_se': recall_se,
            'f1_score_se': f1_score_se
        }

        return metrics

    def calculate_confusion_matrix(self, df_human, df_synthetic, classes, temperature, prompt_type):
        # Creating a confusion matrix
        df_combined = df_human[['key', 'citizen_type']].rename(columns={'citizen_type': 'citizen_type_true'})\
            .merge(df_synthetic[['key', 'citizen_type']].rename(columns={'citizen_type': 'citizen_type_pred'}), on='key')
        df_combined = df_combined[~(df_combined['citizen_type_true'].isin(['unknown', 'defier']))]
        y_true = df_combined['citizen_type_true']
        y_pred = df_combined['citizen_type_pred']
        cm = confusion_matrix(y_true, y_pred, labels=classes)

        # Calculate precision, recall, and f1-score for each class
        precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)

        # Aggregated metrics
        precision_agg, recall_agg, f1_score_agg, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)

        # Store metrics in a dictionary
        metrics = {
            'temperature': temperature,
            'prompt_type': prompt_type,
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1_score': f1_score.tolist(),
            'precision_agg': precision_agg,
            'recall_agg': recall_agg,
            'f1_score_agg': f1_score_agg,
            'cm': cm, 
            'temperature': temperature,
            'prompt_type': prompt_type,
            'n_samples': len(y_true)
        }
        
        return metrics


def plot_class_recalls(df):
    df_ = df.copy()

    # Prepare data for plotting
    df_['recall_never_takers'] = df_['recall'].apply(lambda x: x[1])
    df_['se_recall_never_takers'] = df_['recall_se'].apply(lambda x: x['never taker'])
    
    df_['recall_free_riders'] = df_['recall'].apply(lambda x: x[2])
    df_['se_recall_free_riders'] = df_['recall_se'].apply(lambda x: x['free rider'])
    return df_

def plot_recall(dataframe, recall_type='recall_free_riders', palette='colorblind'):
    """
    Plots specified recall type by 'prompt_type' and 'temperature' with error bars,
    using a color-blind friendly palette.

    Args:
    dataframe (DataFrame): A pandas DataFrame with the required data.
    recall_type (str): The type of recall to plot ('recall_free_riders' or 'recall_never_takers').
    palette (str): The color palette to use. Default is 'colorblind'.

    Returns:
    None: Displays the plot.
    """
    # Validate recall_type
    if recall_type not in ['recall_free_riders', 'recall_never_takers']:
        raise ValueError("recall_type must be either 'recall_free_riders' or 'recall_never_takers'")

    # Standard error column name
    se_column = f'se_{recall_type}'

    # Grouping the data by 'prompt_type' and 'temperature'
    grouped_data = dataframe.groupby(['prompt_type', 'temperature']).agg(
        mean_recall=(recall_type, 'mean'),
        se_recall=(se_column, 'mean')
    ).reset_index()

    # Creating the bar plot with a color-blind friendly palette
    plt.figure(figsize=(8, 5))
    barplot = sns.barplot(
        x='prompt_type', 
        y='mean_recall', 
        hue='temperature', 
        data=grouped_data,
        capsize=0.1,
        palette=palette  # Using the color-blind friendly palette
    )

    # Getting unique values for prompt_type and temperature
    prompt_types = grouped_data['prompt_type'].unique()
    temperatures = grouped_data['temperature'].unique()

    # Iterating over the grouped data to add error bars
    for index, row in grouped_data.iterrows():
        # Calculate the position of the bar
        prompt_index = np.where(prompt_types == row['prompt_type'])[0][0]
        temp_index = np.where(temperatures == row['temperature'])[0][0]
        bar_position = prompt_index * len(temperatures) + temp_index

        # Get the bar object from the barplot
        bar = barplot.patches[bar_position]

        # Get the standard error for the current bar
        se = row['se_recall']
        
        # Get the x coordinate of the bar and the height
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()

        # Draw the error bar
        plt.errorbar(x, y, yerr=se, fmt='none', color='black', capsize=5)

    # Adding titles and labels for clarity
    title = f'{recall_type.replace("_", " ").title()} by Prompt Type and Temperature'
    plt.title(title)
    plt.xlabel('Model & Prompt Type')
    plt.ylabel(f'{recall_type.replace("_", " ").title()}')
    plt.xticks(rotation=90)
    plt.ylim(0,1)

    # Displaying the plot
    plt.tight_layout()
    plt.show()