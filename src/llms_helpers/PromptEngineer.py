import ast
import numpy as np
import pandas as pd
import string
import random
random.seed(42)
import unicodedata

from src.data_helpers.mapper_wuw_meaning import profile_questions
from templates.experiment_prompt import intro_prompt, generate_survey_template_first_round_C2_T2, generate_survey_template_second_round_T2, \
    generate_survey_template_second_round_C2, generate_survey_template_first_round_CA, generate_survey_template_first_round_TA, \
    generate_survey_template_second_round_T1


class SurveyPrompt:
    def __init__(self, df_questions, 
                    df_answers, 
                    num_questions=5, 
                    rand_question_order=False, 
                    max_sampling_attempts=10,
                    survey_context="", 
                    final_question="",
                    final_question_prompt=None,
                    rand_order_options=False,
                    custom_question_ids=[]):
        
        self.df_questions = df_questions
        self.df_answers = df_answers
        self.output_str = ""
        self.num_questions = num_questions
        self.rand_question_order = rand_question_order
        self.max_sampling_attempts = max_sampling_attempts
        self.prompts = []
        self.survey_context = survey_context
        self.final_question = final_question
        self.final_question_prompt = final_question_prompt
        self.rand_order_options = rand_order_options
        self.custom_question_ids = custom_question_ids

    def _normalize_string(self, s):
        return unicodedata.normalize("NFC", s)

    def _unwanted_options(self):
        return ["-1", "-1.0", "nan", "keine Angabe", ""]

    def _unwanted_questions(self):
        return ["Not clear", "Not relevant"]

    # TODO: USE FOR THE FINAL QUESTION? OR NOT NEEDED GIVEN THE EXPERIMENT?
    def _generate_question_prompt(self, question_row):
        question_prompt = f"\nInterviewer: {question_row['Question']}\n"
        if question_row["option_type"] in ["bool", "categorical", "ordinal"]:
            question_prompt += "Bitte wählen Sie nur eine der folgenden Optionen mit dem dazugehörigen Buchstaben aus.\n"
        elif question_row["option_type"] == "numerical":
            question_prompt += "Bitte antworten Sie direkt mit der Nummer.\n"

        return question_prompt

    def _generate_options_prompt(self, question_row):

        # Randomize order if required otherwise show as it is
        if question_row["option_type"] in ["categorical","ordinal",'bool']:
            # Requires to get the option names from question row
            dict_options = ast.literal_eval(question_row['mapping_dict'])
            options = [opt for opt in dict_options if opt not in question_row['exclude_values'].split(",")]
            option_names =  [dict_options[opt] for opt in options]

            if self.rand_order_options:
                indexed_options = list(enumerate(option_names))
                random.shuffle(indexed_options)
                shuffled_indices, shuffled_options = zip(*indexed_options)
                self.option_idx_to_show = list(shuffled_indices)
                self.options_to_show = list(shuffled_options)
            else:
                self.option_idx_to_show = list(enumerate(option_names))
                self.options_to_show = option_names

            # Get alphabet letter for enumerating options
            letters = list(string.ascii_lowercase)
            self.option_strs = [f"({letter}) {opt}" for letter, opt in zip(letters[:len(self.options_to_show)], self.options_to_show)]
            
            if all(len(opt) < 15 for opt in options):
                return " ".join(self.option_strs) + "\n"
            else:
                return "\n".join(self.option_strs) + "\n"
            
        elif question_row["option_type"] == "continuous":
            return ""
        else:
            print(f"ERROR: The data type {question_row['option_type']} is not valid.")
            raise TypeError

    def _generate_answer_prompt(self, question_row, response_row):

        if question_row["option_type"] in ["categorical","ordinal",'bool']:
            # Requires obtainint the answer id from the response row and attribute a name from the question row
            dict_options = ast.literal_eval(question_row['mapping_dict'])
            options = [opt for opt in dict_options if opt not in question_row['exclude_values'].split(",")]
            updated_dict_options = {k: dict_options[k] for k in options if k in dict_options}
            option_names =  [dict_options[opt] for opt in options]

            # Get the answer from the current subject
            response = response_row[question_row['question_id']]

            # Map the answer with the option name if only digits - SHOULD BE THE CASE AS ONLY BUNDESLAND IS STRING
            try:
                if isinstance(response, int):
                    answer_name = updated_dict_options[str(response)]
                else:
                    if response.isdigit():
                        answer_name = updated_dict_options[response]
                    else:
                        print("ERROR: TODO: if answers are not digits a solution should be implemented")
                        raise TypeError
            except:
                print("Skipping qustion as subject did not give a valid answer.")
                return False

            # Map the answer after randomization if any
            index_rand = self.options_to_show.index(answer_name)
            subject_response = self.option_strs[index_rand]

        elif question_row["option_type"] == "continuous":
            subject_response = response_row[question_row['question_id']]
            return f"Ich: NUMFELD {subject_response}\n"

        else:
            print(f"ERROR: The data type {question_row['option_type']} is not valid.")
            raise TypeError
        return f"Ich: {subject_response}\n"

    def _sample_questions(self):
        valid_sampled_questions = pd.DataFrame()
        attempts = 0

        while valid_sampled_questions.shape[0] < self.num_questions and attempts < self.max_sampling_attempts:
            remaining_needed = self.num_questions - valid_sampled_questions.shape[0]
            new_sample = self.df_questions.sample(remaining_needed)
            
            # Exclude unwanted questions from the sample
            new_sample = new_sample[~new_sample["Question"].str.contains('|'.join(self._unwanted_questions()))]

            # Exclude questions with no options provided
            new_sample = new_sample[~new_sample["unique_values"].isna() & (new_sample["unique_values"] != "")]

            valid_sampled_questions = pd.concat([valid_sampled_questions, new_sample])
            valid_sampled_questions = valid_sampled_questions[~valid_sampled_questions["Question"].isin(self._unwanted_options())]
            attempts += 1

        if self.rand_question_order:
            valid_sampled_questions = valid_sampled_questions.sample(frac=1)  # shuffle the order

        return valid_sampled_questions
    
    def _survey_context_custom_subject(self, series):
        profile_answers = {'bundesland_name': series['bundesland_name'],
                          'altq': series['altq']}
        # TODO: add alternative processing if continuous variables added to the profile
        df_q_profile = self.df_questions[self.df_questions['question_id'].isin(profile_questions)]
        for question in profile_questions:
            if question not in ['bundesland_name','altq']:
                id_value = series[question]
                unwanted_series = df_q_profile[df_q_profile['question_id']==question]['exclude_values']
                if len(unwanted_series)>0:
                    unwanted_values = df_q_profile[df_q_profile['question_id']==question]['exclude_values'].values[0]
                else:
                    unwanted_values = []
                if question == 'ges':
                    mapper = {'1': ['','r','Mann'], '2':['e','','Frau']}
                elif question == 'city_category':
                    mapper = {'1':'weniger als 20.000 Einwohnern','2':'zwischen 20.000 bis 100.000 Einwohnern','3':'mehr als 100.000 Einwohnern'}
                elif question == 'so1':
                    mapper = {'-1': 'Keine Angabe', '1': 'Kein Abschluss', '2': 'Abschluss nach höchstens 7 Jahren Schule', '3': 'Haupt-/Volksschulabschluss', '4': 'Realschulabschluss (Mittlere Reife)', '5': 'Fachhochschulreife', '6': 'Abitur'}
                elif question == "so5":
                    mapper = {'-1': 'Keine Angabe', '1': 'unter 700 Euro', '2': '700 bis unter 1.200 Euro', '3': '1.200 bis unter 1.700 Euro', '4': '1.700 bis unter 2.200 Euro', '5': '2.200 bis unter 2.700 Euro', '6': '2.700 bis unter 3.200 Euro', '7': '3.200 bis unter 3.700 Euro', '8': '3.700 bis unter 4.200 Euro', '9': '4.200 bis unter 4.700 Euro', '10': '4.700 bis unter 5.200 Euro', '11': '5.200 bis unter 5.700 Euro', '12': '5.700 Euro und mehr','13':'Weiß nicht/keine Angabe'}                
                elif question == 'so2':
                    # artificial mapping
                    mapper = {'-1': 'Sekundarabschluss I oder darunter', '1': 'Sekundarabschluss I oder darunter', '2': 'Anlernausbildung/Berufspraktikum (12 Monate)', '3': 'Berufsvorbereitungsjahr', '4': 'Lehre, Berufsausbildung im dualen System', '5': 'Vorbereitungsdienst für den mittleren Dienst in der öffentlichen Verwaltung', '6': 'Berufsabschluss (Berufsfachschule/Kollegschule, 1-jährige Gesundheitsschule)', '7': '2/3-jährige Gesundheitsschule (z.B. Krankenpflege)', '8': 'ein Fachschulabschluss (Meister/-in, Techniker/-in oder gleichwertiger Abschluss)', '9': 'Berufsakademie, Fachakademie', '10': 'Abschluss einer Verwaltungsfachhochschule', '11': 'Fachhochschulabschluss, auch Ingenieurschulabschluss', '12': 'Abschluss einer Universität, wissenschaftlichen Hochschule, Kunsthochschule', '13': 'Promotion','14': 'Sekundarabschluss I oder darunter'}
                elif question == 'ist6':
                    mapper = {'1':'Bis 1918','2':'1919 bis 1948','3':'1949 bis 1957','4':'1958 bis 1968','5':'1969 bis 1978','6':'1979 bis 1983','7':'1984 bis 1994','8':'1995 bis 2001','9':'2002 bis 2004','10':'2002 bis 2004','11':'2005 bis 2006','12':'2007 bis 2008','13':'2007 bis 2008','14':'2007 bis 2008','15':'2009 bis 2013','16':'2014 bis 2015','17':'2016 bis 2019','18':'Ab 2020'}
                else:
                    mapper = ast.literal_eval(df_q_profile[df_q_profile['question_id']==question]['mapping_dict'])
                if unwanted_values != np.nan or unwanted_values != None or unwanted_values != '':
                    if isinstance(unwanted_values, float):
                        profile_answers[question] = mapper[str(id_value)]
                    else:
                        unwanted_values = unwanted_values.split(",")
                        profile_answers[question] = mapper[str(id_value)] if id_value not in unwanted_values else None
                else:
                    profile_answers[question] = mapper[str(id_value)]

        return intro_prompt(profile_answers)

    def _custom_sample_questions_ids(self, completed_q_ids):
        # Flatten the list of lists keeping track of origin.
        all_items = [(item, idx) for idx, lst in enumerate(self.rand_question_order) for item in lst if item in completed_q_ids]
        
        # Shuffle the items to ensure randomness.
        random.shuffle(all_items)
        
        # Initialize count for each list.
        counts = [0] * len(self.rand_question_order)
        
        # Initialize the result list.
        sampled_items = []
        
        # Keep track of the indices we have used from each list.
        used_indices = {i: set() for i in range(len(self.rand_question_order))}
        
        for _ in range(self.num_questions):
            for list_idx, _ in enumerate(self.rand_question_order):
                # Skip if this list is already fully used.
                if counts[list_idx] >= len(self.rand_question_order[list_idx]):
                    continue
                
                # Find the next unused item from this list.
                for item, origin_idx in all_items:
                    if origin_idx == list_idx and item not in used_indices[list_idx]:
                        sampled_items.append(item)
                        used_indices[list_idx].add(item)
                        counts[list_idx] += 1
                        break
                
                # Check if we've sampled enough.
                if len(sampled_items) == self.num_questions:
                    assert len(sampled_items) == len(set(sampled_items))
                    return sampled_items
        
        # If we've gone through all lists and haven't filled the quota,
        # fill the rest with repeated items.
        while len(sampled_items) < self.num_questions:
            for list_idx, lst in enumerate(self.rand_question_order):
                # We'll add items from lists that are not yet exhausted.
                if counts[list_idx] < len(lst):
                    for item in lst:
                        if item not in used_indices[list_idx]:
                            sampled_items.append(item)
                            used_indices[list_idx].add(item)
                            counts[list_idx] += 1
                            break
                
                # Check if we've sampled enough.
                if len(sampled_items) == self.num_questions:
                    assert len(sampled_items) == len(set(sampled_items))
                    return sampled_items

        assert len(sampled_items) == len(set(sampled_items))
        return sampled_items
    
    def _custom_sample_questions(self, row_responses):
        # get a mapper of question_id: unwanted responses
        unwanted_dict = dict(zip(self.df_questions[self.df_questions['option_type']!='continuous']['question_id'], 
                                 self.df_questions[self.df_questions['option_type']!='continuous']['exclude_values']))
        
        # get question_ids with non empty responses
        completed_q_ids = []
        for q_id in list(row_responses.index):
            try:
                unwanted_values = unwanted_dict[q_id].split(",")
                if str(row_responses[q_id]) not in unwanted_values:
                    completed_q_ids.append(q_id)
            except:
                completed_q_ids.append(q_id)

        if len(self.custom_question_ids) == 0:
            question_ids = self._custom_sample_questions_ids(completed_q_ids)
        else:
            question_ids = self.custom_question_ids
        return self.df_questions[self.df_questions['question_id'].isin(question_ids)]

    def _custom_final_question(self, series):
        EBJ = series['ebj']
        EBE = series['ebe']
        EBU = series['ebu']
        KDJ = series['kdj']
        KDE = series['kde']
        KDU = series['kdu']
        KDF = series['kdf']
        ist_5 = series['ist5']

        if self.final_question_prompt == 'C2_T2':
            return generate_survey_template_first_round_C2_T2(EBJ, EBE, EBU)
        
        elif self.final_question_prompt == 'T2':
            return generate_survey_template_second_round_T2(ist_5, EBJ, EBE, EBU, KDJ, KDE, KDU, KDF)
        
        elif self.final_question_prompt == 'C2':
            return generate_survey_template_second_round_C2(EBJ, EBE, EBU)
        
        elif self.final_question_prompt == 'CA':
            return generate_survey_template_first_round_CA(EBJ, EBE, EBU)
        
        elif self.final_question_prompt == 'TA':
            return generate_survey_template_first_round_TA(ist_5, EBJ, EBE, EBU, KDJ, KDE, KDU, KDF)
        
        elif self.final_question_prompt == 'T1':
            return generate_survey_template_second_round_T1(ist_5, EBJ, EBE, EBU, KDJ, KDE, KDU, KDF)
        
        else:
            print(f"{self.final_question_prompt} is not a known final question prompt.")
            raise NameError

    def generate(self):
        # the answers are previously filtered per subject (key)
        self.question_ids = []
        for _, response_row in self.df_answers.iterrows():
            if self.survey_context != "custom_subject":
                subject_prompt = f"{self.survey_context}\n"
            else:
                subject_prompt = self._survey_context_custom_subject(response_row)

            if self.rand_question_order in [True, False]:
                sampled_questions = self._sample_questions()
            else:
                sampled_questions = self._custom_sample_questions(response_row)

            self.question_ids.append(sampled_questions['question_id'].tolist())

            for _, question_row in sampled_questions.iterrows():
                questions = self._generate_question_prompt(question_row)
                options = self._generate_options_prompt(question_row)
                answers = self._generate_answer_prompt(question_row, response_row)
                if answers != False:
                    subject_prompt += questions
                    subject_prompt += options
                    subject_prompt += answers

            if self.final_question != "custom_subject":
                subject_prompt += self.final_question
            else:
                subject_prompt += self._custom_final_question(response_row)
            self.prompts.append(subject_prompt)


    def display(self):
        for prompt in self.prompts:
            print(prompt)
            print("----------")
        return self.prompts
