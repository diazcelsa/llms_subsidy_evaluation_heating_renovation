import numpy as np
import numpy_financial as npf
import pandas as pd

import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import matplotlib.pyplot as plt

class DeltaNVP:
    def __init__(self, discount_rate=3.62):
        # https://www.bundesbank.de/en/press/press-releases/announcement-of-the-basic-rate-of-interest-as-of-1-january-2024-adjustment-to-3-62--920962
        self.r = discount_rate
        self.cost_Aj = 300
        self.npv_Aj = None
        self.npv_Bj = None

    def set_parameters(self, row, opportunity_cost):
        self.S_Aj = row['kdj'] - row['kde']
        self.S_Bj = row['kdj'] - row['kdu']
        self.Tj = row['a6']
        self.beta = 0.71 * row['ist5']
        self.eps_original = row['ebj']
        self.eps_improved_Aj = row['ebe']
        self.eps_improved_Bj = row['ebu']
        self.opportunity_cost = opportunity_cost

        # If these values are expected to be single values in a Series, use .values[0]
        # Otherwise, if they are single values already, this is not needed.
        if isinstance(self.S_Aj, pd.Series):
            self.S_Aj = self.S_Aj.values[0]
        if isinstance(self.S_Bj, pd.Series):
            self.S_Bj = self.S_Bj.values[0]
        if isinstance(self.Tj, pd.Series):
            self.Tj = self.Tj.values[0]
        if isinstance(self.beta, pd.Series):
            self.beta = self.beta.values[0]
        if isinstance(self.eps_original, pd.Series):
            self.eps_original = self.eps_original.values[0]
        if isinstance(self.eps_improved_Aj, pd.Series):
            self.eps_improved_Aj = self.eps_improved_Aj.values[0]
        if isinstance(self.eps_improved_Bj, pd.Series):
            self.eps_improved_Bj = self.eps_improved_Bj.values[0]

    def calculate_npv_Aj(self):
        gain_market_value = self.beta * (self.eps_original - self.eps_improved_Aj)
        cashflows = [self.S_Aj for _ in range(self.Tj)]
        self.npv_Aj = npf.npv(self.r, cashflows) + gain_market_value - self.cost_Aj

    def calculate_npv_Bj(self, j):
        gain_market_value = self.beta * (self.eps_original - self.eps_improved_Bj)
        cashflows = [self.S_Bj for _ in range(self.Tj)]
        self.npv_Bj = npf.npv(self.r, cashflows) + gain_market_value - (self.cost_Aj + self.opportunity_cost[j])

    def transform_options_to_probabilities(self, df):
        df_ = df.copy()
        for col in df.columns:
            df_[col] = df_[col].map({1: 0, 2: 1})
        return df_
    
    def optimal_utility(self, decisions):
        delta_npvs = []
        self.calculate_npv_Aj()
        decisions_ = self.transform_options_to_probabilities(decisions)
        for j, decision in enumerate(decisions_.columns):
            self.calculate_npv_Bj(j)
            npvs = [self.npv_Aj, self.npv_Bj]
            objective_utility_j = npvs.index(max(npvs))
            subjective_utility_j = decisions_.iloc[0][decision]
            delta_npv = 0
            # risk taker, choose B despite of A more profitable
            if int(subjective_utility_j) > int(objective_utility_j):
                delta_npv = np.abs(npvs[int(subjective_utility_j)] - npvs[int(objective_utility_j)])
            # risk adverse, choose A depite of B more profitable
            elif int(subjective_utility_j) < int(objective_utility_j):
                delta_npv = np.abs(npvs[int(subjective_utility_j)] - npvs[int(objective_utility_j)]) * (-1)

            decision_info = {
                'j': j,
                'npv_Aj': self.npv_Aj,
                'npv_Bj': self.npv_Bj,
                'objective_utility': objective_utility_j,
                'subjective_utility': subjective_utility_j,
                'ANPV': delta_npv
            }
            delta_npvs.append(decision_info)
        return delta_npvs

    def optimal_utility_legacy(self, decisions):
        delta_npvs = []
        self.calculate_npv_Aj()
        decisions_ = self.transform_options_to_probabilities(decisions)
        for j, decision in enumerate(decisions_.columns):
            self.calculate_npv_Bj(j)
            npvs = [self.npv_Aj, self.npv_Bj]
            objective_utility_j = npvs.index(max(npvs))
            subjective_utility_j = decisions_.iloc[0][decision]
            delta_npv = npvs[int(subjective_utility_j)] - npvs[int(objective_utility_j)]
            decision_info = {
                'j': j,
                'npv_Aj': self.npv_Aj,
                'npv_Bj': self.npv_Bj,
                'objective_utility': objective_utility_j,
                'subjective_utility': subjective_utility_j,
                'ANPV': delta_npv
            }
            delta_npvs.append(decision_info)
        return delta_npvs
    
    def calculate_deltas(self, df, opportunity_cost, final_columns):
        key_ids = df['key'].unique()
        deltas = []
        for key in key_ids:
            row = df[df['key'] == key].iloc[0]  # Assuming that the 'key' identifies a unique row
            self.set_parameters(row, opportunity_cost)
            delta_npvs = self.optimal_utility(df.loc[df['key'] == key, final_columns])
            df_deltas = pd.DataFrame(delta_npvs)
            df_deltas['key'] = key
            df_deltas['citizen_type'] = row['citizen_type']
            df_deltas['opp_cost'] = opportunity_cost
            deltas.append(df_deltas)
        df_deltas_combined = pd.concat(deltas)
        self.plot_violin_grid(df_deltas_combined)
        return df_deltas_combined
    
    def plot_violin_grid(self, dataframe, col_wrap=4, height=3, aspect=1.5, rotation=45):
        """
        Plots a grid of violin plots for each value of 'j' in the DataFrame.

        :param dataframe: A pandas DataFrame with 'j', 'citizen_type', and 'ANPV' columns.
        :param col_wrap: Number of columns before wrapping to the next row.
        :param height: Height (in inches) of each facet.
        :param aspect: Aspect ratio of each facet, so that aspect * height gives the width of each facet in inches.
        :param rotation: Degrees to rotate the x-tick labels.
        """
        # Set up the FacetGrid with one plot per row and specified dimensions for visibility
        g = sns.FacetGrid(dataframe, col="j", col_wrap=col_wrap, height=height, aspect=aspect, sharey=False, sharex=False)

        # Map the sns.violinplot to the grid
        g.map(sns.violinplot, "citizen_type", "ANPV", palette="Set2", cut=0)

        # Set the y-axis labels and x-tick labels
        for ax in g.axes:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation)  # Rotate x-tick labels for visibility

        # Add a legend for citizen type
        g.add_legend(title="Citizen Type")

        # Adjust the layout to prevent overlapping
        plt.tight_layout()

        # Show the plot
        plt.show()
    
    def prepare_for_ols_logistic_regression(self, df, bool_cols, cat_cols):
        df_ = df.copy()
        # Convert boolean columns to 0/1
        for col in bool_cols:
            df_[col] = df_[col].astype(int)

        # Dummify categorical columns
        for col in cat_cols:
            dummies = pd.get_dummies(df_[col], prefix=col, dtype=int)
            df_ = pd.concat([df_, dummies], axis=1)
            df_.drop(col, axis=1, inplace=True)

        return df_
    
    def create_boolean_columns(self,df):
        df_ = df.copy()
        df_['is_low_income'] = df_['total_income_level'].isin(['low_income', 'poverty']).astype(int)
        df_['is_high_wealth'] = df_['is_high_wealth'].astype(int)
        df_['is_secondary_school_highest'] = (df_['high_education_level'] == 'secondary_school').astype(int)
        df_['is_right_oriented'] = (df_['pol_orientation'] == 'right').astype(int)
        df_['is_altruist'] = (df_['is_altruist'] == True).astype(int)  #
        df_['is_nature_protector'] = df_['nature_level'].isin(['High', 'Medium']).astype(int)
        df_['is_short_term_profit'] = (df_['profit_focus'] == 'short').astype(int)
        df_['is_psychologically_capable'] = df_['ownership_level'].isin(['High', 'Medium']).astype(int) 
        df_['ee_belief'] = (df_['ee_belief'] == True).astype(int)
        return df_
    
    def add_interaction_factors(self, df):
        # Define the columns for which to create interaction terms with 'opp_cost'
        interaction_columns = [
            'is_psychologically_capable','ee_belief','is_nature_protector'
        ]
        # Create interaction terms with 'opp_cost'
        new_cols = []
        for col in interaction_columns:
            interaction_term = col + "_opp_cost_interaction"
            new_cols.append(interaction_term)
            df[interaction_term] = df[col] * df['opp_cost']
        return df, new_cols
    
    def model_delta_nvp(self, result_df, df,  dependent_var, independent_vars):
        # Set the entity identifier as the index
        df_ = self.create_boolean_columns(df)
        df_ = df_.set_index('key')
        df_result = result_df[['key','opp_cost',dependent_var]].set_index('key')
        merged_df = pd.merge(df_, df_result, left_index=True, right_index=True)
        merged_df, new_cols = self.add_interaction_factors(merged_df)
        scaler = StandardScaler()
        merged_df['std_age'] = scaler.fit_transform(merged_df[['altq']])

        # Create a PanelOLS model with entity effects
        independent_vars = [i for i in independent_vars if i != 'altq']# + ['opp_cost'] #'std_age',
        exog = sm.add_constant(merged_df[independent_vars + new_cols])
        endog = scaler.fit_transform(merged_df[[dependent_var]])
        mod = sm.OLS(endog, exog)

        # Fit the model
        res = mod.fit()

        # Collect model information, including coefficients
        return res.summary(), merged_df
