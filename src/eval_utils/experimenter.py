
import os
import re
import sys
import ast
sys.path.append("../../../")

import time
import pandas as pd
import numpy as np
from datetime import datetime
from collections import namedtuple

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches


from src.llms_helpers.openai_api import query_open_ai
from src.llms_helpers.PromptEngineer import SurveyPrompt
from src.eval_utils.responses_parser import save_prompt_to_file


Data = namedtuple('Data', ['questions', 'answers', 'custom_question_ids'])
ModelConfig = namedtuple('ModelConfig', ['model', 'n_trials'])
SurveyConfig = namedtuple('SurveyConfig', ['n_iter', 'num_questions', 'rand_question_order', 'survey_context',
                                           'final_question', 'final_question_prompt', 'rand_order_options'])
EvalConfig = namedtuple('FinalConfig', ['mapper', 'final_columns'])

class ExperimentRunner:
    def __init__(self, data, model_config, survey_config, eval_config):

        self.df_questions, self.df_answers, self.custom_question_ids = data
        self.model, self.n_trials = model_config
        self.n_iter, self.num_questions, self.rand_question_order, self.survey_context, self.final_question, \
            self.final_question_prompt, self.rand_order_options = survey_config
        self.mapper, self.final_columns = eval_config
        self.generated_prompts = []

    def generate_prompts(self):
        self.survey = SurveyPrompt(self.df_questions, 
                              self.df_answers, 
                              num_questions=self.num_questions, 
                              rand_question_order=self.rand_question_order, 
                              survey_context=self.survey_context,
                              final_question=self.final_question,
                              final_question_prompt=self.final_question_prompt,
                              rand_order_options=self.rand_order_options,
                              custom_question_ids=self.custom_question_ids
                             )
        self.survey.generate()
        self.generated_prompts = self.survey.display()

    def store_prompts(self, directory, new_prompts=[]):
        # Check if the directory exists, create it if it doesn't
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Define the configuration for saving prompts
        save_prompt_config = {
            "group": self.final_question_prompt,
            "order": "A",
            "is_random": self.rand_order_options,
            "n_questions": self.num_questions-1,
        }

        # List to store configurations for each prompt
        updated_prompt_configs = []

        # Iterate over each generated prompt
        if len(new_prompts)==0:
            for i, prompt in enumerate(self.generated_prompts):
                # Update the save prompt configuration
                updated_save_prompt_config = save_prompt_to_file(prompt, self.df_answers['key'].iloc[i], self.survey.question_ids[i], save_prompt_config, directory)
                updated_prompt_configs.append(updated_save_prompt_config)
        else:
            for i, prompt in enumerate(new_prompts):
                # Update the save prompt configuration
                updated_save_prompt_config = save_prompt_to_file(prompt, self.df_answers['key'].iloc[i], self.survey.question_ids[i], save_prompt_config, directory)
                updated_prompt_configs.append(updated_save_prompt_config)

    def extract_data(self, string, mapper):
        # Patterns to match:
        # "1. (A)", "1. A", "1. Option A", "1. (Option A)"
        # "1. (B)" (and similar formats for other letters)
        # "1. (A)", "1. A", "1. (a)", "1. a"
        # "1. Option A", "1. Option B", "1. Option a", "1. Option b"
        # Including patterns like "1. (B) 2. (B) 3. (B) ..."
        simple_pattern = r'(\d+)\.\s\(([A-Za-z])\)'
        matches = re.findall(simple_pattern, string)

        # If no matches are found, try the more complex pattern
        if len(matches) == 0:
            complex_pattern = r'(\d+)\.\s(?:\(?Option\s)?([A-Za-z])\)?(?:\s|\n|$)'
            matches = re.findall(complex_pattern, string)

            if len(matches) == 0:
                raise ValueError("No matches found")
        
        if len(matches) < 15:
            raise ValueError("Number of matches below 15")

        # Convert letters to uppercase if they are in lowercase
        result = {int(number): mapper[letter.upper()] for number, letter in matches}
        return result
    
    def extract_data_from_openai_object_gpt_babbage_002(self, response_object):
        data = {
            "created": response_object.created,
            "id": response_object.id,
            "model": response_object.model,
            "object": response_object.object,
        }

        # Extract fields from 'choices'
        # Assuming you're interested in 'text' and 'finish_reason'
        for i, choice in enumerate(response_object.choices):
            data[f"choice_{i}_finish_reason"] = choice.finish_reason
            data[f"choice_{i}_text"] = choice.text
            

        # Flatten 'usage' fields
        data.update({
            "completion_tokens": response_object.usage.completion_tokens,
            "prompt_tokens": response_object.usage.prompt_tokens,
            "total_tokens": response_object.usage.total_tokens
        })

        return data

    def extract_data_from_openai_object_gpt_3_5_turbo(self, response_object):
        # Extracting data directly from OpenAIObject attributes
        data = {
            "finish_reason": response_object.choices[0].finish_reason,
            "index": response_object.choices[0].index,
            "message_content": response_object.choices[0].message.content,
            "message_role": response_object.choices[0].message.role,
            "created": response_object.created,
            "id": response_object.id,
            "model": response_object.model,
            "object": response_object.object,
            "completion_tokens": response_object.usage.completion_tokens,
            "prompt_tokens": response_object.usage.prompt_tokens,
            "total_tokens": response_object.usage.total_tokens,
            "completion_tokens": response_object.usage.completion_tokens,
            "prompt_tokens": response_object.usage.prompt_tokens,
            "total_tokens": response_object.usage.total_tokens
        }

        return data

    def extract_data_from_openai_object(self, response_object, model="babbage-002"):
        # Check if the response_object is already in the correct format
        # if isinstance(response_object, dict):
        #     data = response_object
        # else:
        # Extract data based on model
        if model == "gpt-3.5-turbo":
            data = self.extract_data_from_openai_object_gpt_3_5_turbo(response_object)
        elif model == "gpt-4-1106-preview" or model == "gpt-3.5-turbo-1106":
            data = self.extract_data_from_openai_object_gpt_3_5_turbo(response_object)
        elif model == "babbage-002":
            data = self.extract_data_from_openai_object_gpt_babbage_002(response_object)
        else:
            raise ValueError("Unsupported model type")

        # Create DataFrame from the extracted data
        df = pd.DataFrame([data])

        # Extract 'text' and 'finish_reason' from 'choices', if they exist
        if 'choices' in df.columns and model == "babbage-002":
            df['text'] = df['choices'].apply(lambda x: x[0]['text'] if x and 'text' in x[0] else None)
            df['finish_reason'] = df['choices'].apply(lambda x: x[0]['finish_reason'] if x and 'finish_reason' in x[0] else None)
            columns_to_drop = ['choices', 'usage']  # Add other columns if needed
            df = df.drop(columns=columns_to_drop) 
        if model == "gpt-3.5-turbo" or model == "gpt-4-1106-preview":
            # Extract 'text' and 'finish_reason' from 'choices'
            if 'choices' in df.columns:
                df['text'] = df['choices'].apply(lambda x: x[0]['message']['content'] if x and 'message' in x[0] and 'content' in x[0]['message'] else None)
                df['finish_reason'] = df['choices'].apply(lambda x: x[0]['finish_reason'] if x and 'finish_reason' in x[0] else None)
                df['role'] = df['choices'].apply(lambda x: x[0]['message']['role'] if x and 'message' in x[0] and 'role' in x[0]['message'] else None)

            # Extract fields from 'usage'
            if 'usage' in df.columns:
                usage_fields = ['prompt_tokens', 'completion_tokens', 'total_tokens']
                for field in usage_fields:
                    df[field] = df['usage'].apply(lambda x: x[field] if x and field in x else None)

            if 'message_content' in df.columns:
                return df

            # Drop the original nested columns
            columns_to_drop = ['choices', 'usage']
            df = df.drop(columns=columns_to_drop)
            
        return df

    def clean_and_convert_to_dict(input_string, mapper):
        # Remove unwanted characters and split into lines
        cleaned_lines = input_string.replace("'", "").split('\n')

        # Initialize an empty dictionary
        result_dict = {}

        # Track enumeration for values not in mapper
        enumeration_counter = max(mapper.values()) + 1 if mapper else 1

        # Extract integers and corresponding vowels, and add to the dictionary
        for line in cleaned_lines:
            parts = line.split('. ')
            if len(parts) == 2:
                key = int(parts[0])  # Convert to integer
                value = parts[1].strip(')').strip()  # Remove the closing parenthesis and trim whitespaces

                # If the value is not in mapper, assign a new enumeration
                if value not in mapper:
                    mapper[value] = enumeration_counter
                    enumeration_counter += 1

                result_dict[key] = mapper[value]

        return result_dict
    
    def query_and_extract(self, prompt, temperature, top_p, max_tokens):
        message, response = query_open_ai(prompt, model_name=self.model, max_tokens=max_tokens, 
                                            temperature=temperature, top_p=top_p)
        df_response = self.extract_data_from_openai_object(response, model=self.model)
        return message, True

    def process_and_store(self, index, extracted_result, iter, temperature, output_file):
        df = pd.DataFrame(columns=self.final_columns)
        for col_index, value in extracted_result.items():
            df.loc[0, self.final_columns[col_index-1]] = value
        df['key'] = self.df_answers['key'].iloc[index]
        df['label'] = "synthetic"
        df['temperature'] = temperature
        df['iter'] = iter
        df['success'] = True
        with open(output_file, mode='a' if os.path.exists(output_file) else 'w', newline='', encoding='utf-8') as file:
            df.to_csv(file, header=not file.tell(), index=False)

    def prepare_output_directory(self, output_file):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_dir = os.path.join(os.path.dirname(output_file), timestamp)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        return output_dir, os.path.join(output_dir, os.path.basename(output_file))

    def run_experiment(self, output_file, temperature, top_p, max_tokens):
        output_dir, output_file_with_timestamp = self.prepare_output_directory(output_file)

        for i, prompt in enumerate(self.generated_prompts):
            for iter in range(self.n_iter): 
                print(f"Iteration {iter} for prompt index {i}.")
                retry_count = 0
                is_success = False

                while not is_success and retry_count < self.n_trials:
                    try:
                        message, is_success = self.query_and_extract(prompt, temperature, top_p, max_tokens)
                        extracted_result = self.extract_data(message, self.mapper) #df_response['text'][0], self.mapper)
                        print("Success!!")
                    except Exception as e:
                        print(f"Error or incomplete response: {e}")
                        retry_count += 1
                        time.sleep(5)

                if is_success:
                    self.process_and_store(i, extracted_result, iter, temperature, output_file_with_timestamp)

        print(f"Results stored in {output_file_with_timestamp}")

    def run_recurrent_experiment(self, output_file, temperature, top_p, max_tokens, additional_sentence):
        output_dir, output_file_with_timestamp = self.prepare_output_directory(output_file)
        new_prompts = []

        for i, initial_prompt in enumerate(self.generated_prompts):
            current_prompt = initial_prompt
            try:
                message, success = self.query_and_extract(current_prompt, temperature, top_p, max_tokens * 7)
                if success:
                    print("Initial thinking done.")
                    current_prompt += "\n" + message + "\n" + additional_sentence # df_response['text'][0] + "\n" + additional_sentence
                    new_prompts.append(current_prompt)
                else:
                    print("Initial thinking not correctly extracted.")
            except Exception as e:
                time.sleep(5)
                print(f"Error in initial response: {e}")

            for iter in range(self.n_iter):
                print(f"Iteration {iter} for prompt index {i}.")
                retry_count = 0
                is_success = False
                extracted_result = {} 


                while not is_success and retry_count < self.n_trials:
                    try:
                        message, is_success = self.query_and_extract(current_prompt, temperature, top_p, max_tokens)
                        if is_success:
                            extracted_result = self.extract_data(message, self.mapper) # df_response['text'][0], self.mapper)
                            self.process_and_store(i, extracted_result, iter, temperature, output_file_with_timestamp)
                            print("Success!!")
                    except Exception as e:
                        print(f"Error or incomplete response: {e}")
                        retry_count += 1
                        is_success = False  # Ensure is_success is set to False in case of an exception
                        time.sleep(5)

                    if retry_count == 0 and is_success is False:
                        print("modify prompt further")
                        current_prompt += "\n" + "Interviewer: Halten Sie Ihre Antworten in einer nummerierten Liste organisiert: '1. (Option Buchstabe) 2. (Option Buchstabe) 3. (Option Buchstabe) usw." + "\n" + "Ich: "
   
        print(f"Results stored in {output_file_with_timestamp}")

        output_file_with_timestamp_prompt = output_dir + f'/prompts_{temperature}'
        self.store_prompts(directory=output_file_with_timestamp_prompt, new_prompts=new_prompts)
        print(f"Prompts stored in {output_file_with_timestamp_prompt}")

    def ensure_numeric(self, df):
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.dropna(axis=1, how='all')
    
    def plot_individual_heatmaps(self, dfs, titles, config=''):
        # Set the font size for labels, titles, and legend
        plt.rcParams.update({'font.size': 18})

        # Define the colors and labels for each unique value
        value_colors = {1: '#1f77b4', 2: '#ff7f0e'}  # Blue for 1, Orange for 2
        value_labels = {1: 'Simple Renovation', 2: 'Complex Renovation'}

        n_heatmaps = len(dfs)
        fig, axes = plt.subplots(1, n_heatmaps, figsize=(12 * n_heatmaps, 20), sharex=True)

        for i, df in enumerate(dfs):
            if 'key' in df.columns and df.index.name != 'key':
                df = df.set_index('key')

            # Determine which values are present in the dataframe
            unique_values = np.unique(df.values)
            cmap = ListedColormap([value_colors[val] for val in unique_values if val in value_colors])

            sns.heatmap(df, annot=False, cmap=cmap, cbar=False, ax=axes[i] if n_heatmaps > 1 else axes)
            axes[i if n_heatmaps > 1 else None].set_title(titles[i], fontsize=18)
            axes[i if n_heatmaps > 1 else None].set_yticklabels(df.index, rotation=0, fontsize=12)

        # Create legend patches
        patches = [mpatches.Patch(color=value_colors[val], label=value_labels[val]) 
                   for val in unique_values if val in value_colors]

        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=18)

        if config:
            try:
                additional_config = ast.literal_eval(config)
                plt.rcParams.update(additional_config)
            except ValueError:
                print("Invalid configuration string")

        plt.tight_layout()
        plt.show()

    def get_sort_order(self, df):
        # Sort the dataframe by values and return the order of keys
        df_sorted = df.set_index('key')
        counts = df_sorted.apply(pd.Series.value_counts, axis=1).fillna(0)
        sort_order = sum(counts[cat] * (10 ** (len(counts.columns) - idx)) for idx, cat in enumerate(counts.columns))
        counts['sort_order'] = sort_order
        sorted_indices = counts.sort_values(by='sort_order', ascending=False).index
        return sorted_indices

    def compare_results(self, df_human_sample, df_synthetic_samples):
        # Filter and ensure numeric data
        df_synthetic_samples_filtered = self.ensure_numeric(df_synthetic_samples)
        df_human_sample_numeric = self.ensure_numeric(df_human_sample)

        # Get sort order based on the human sample dataframe
        human_sort_order = self.get_sort_order(df_human_sample_numeric)

        # Apply the same sort order to both human and synthetic samples
        df_synthetic_samples_sorted = df_synthetic_samples_filtered.set_index('key').reindex(human_sort_order).reset_index()
        df_human_sample_sorted = df_human_sample_numeric.set_index('key').reindex(human_sort_order).reset_index()

        # Prepare dataframes for plotting
        titles = ["Synthetic Samples", "Human Samples"]
        dfs = [df_synthetic_samples_sorted[['key'] + self.final_columns],
               df_human_sample_sorted[['key'] + self.final_columns]]

        # Plot the heatmaps
        self.plot_individual_heatmaps(dfs, titles)

    