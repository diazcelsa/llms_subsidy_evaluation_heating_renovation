import os
import json
import pandas as pd

def save_prompt_to_file(prompt, key, question_ids, save_prompt_config, directory="prompts"):
    """
    Saves a given prompt to a text file with a filename encoding specified identifiers.

    :param prompt: The text of the prompt to be saved.
    :param identifier1: The first identifier to include in the filename.
    :param identifier2: The second identifier to include in the filename.
    :param directory: The directory where the file will be saved. Default is 'prompts'.
    """

    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Create filename
    filename = f"{key}_{save_prompt_config['group']}_{save_prompt_config['order']}_{save_prompt_config['is_random']}_{save_prompt_config['n_questions']}.txt"

    # Full path for the file
    filepath = os.path.join(directory, filename)

    prompt_series = pd.Series(save_prompt_config)
    prompt_series['filepath'] = filepath
    prompt_series['key'] = key
    prompt_series['question_ids'] = question_ids

    # Write the prompt to the file
    with open(filepath, 'w', encoding='utf-8') as file:
        file.write(prompt)
    print(f"Prompt saved to {filepath}")
    return prompt_series

def extract_data_from_openai_object_gpt_babbage_002(response_object):
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

def extract_data_from_openai_object_gpt_3_5_turbo(response_object):
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

def extract_data_from_openai_object(response_object, model="babbage-002"):
    # Check if the response_object is already in the correct format
    if isinstance(response_object, dict):
        data = response_object
    else:
        # Extract data based on model
        if model == "gpt-3.5-turbo":
            data = extract_data_from_openai_object_gpt_3_5_turbo(response_object)
        elif model == "babbage-002":
            data = extract_data_from_openai_object_gpt_babbage_002(response_object)
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
    if model == "gpt-3.5-turbo":
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