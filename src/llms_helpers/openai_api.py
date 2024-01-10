import os
import openai

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(dotenv_path="../../../secrets_config.env")

# TODO: USE gpt-3.5-turbo-instruct
def query_open_ai(prompt, model_name="babbage-002", max_tokens=150, temperature=0.7, top_p=5, 
                  list_messages=[{"role": "assistant", "content": ""}, {"role": "user", "content": ""}]):
    """
    Query the GPT-3 model with a given prompt.
    
    Parameters:
    - prompt (str): The input string to send to the model.
    - model_name (str): Which GPT-3 model version to use. Default is "text-davinci-002".
    - max_tokens (int): Maximum length of the output.
    - temperature (float): Determines the randomness of the model's output.
    
    Returns:
    - str: The model's response.
    """

    # Ensure your API key is set
    openai.api_key = "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" #"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"#os.getenv('OPENAI_API_KEY')
    
    if not openai.api_key:
        raise ValueError("API key not found. Ensure it's set in your secrets_config.env file.")
    
    # Check if using a chat model
    if "turbo" in model_name:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature
        )
        # Extract the model's response from the API response
        print(response)
        message = response.choices[0].message["content"].strip()

    elif "gpt-4" in model_name or "gpt-3.5-turbo-1106" in model_name:
        print("went into gpt-4")
        response = openai.ChatCompletion.create(
        model=model_name,
        max_tokens=max_tokens,
        top_p=top_p,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}])
        # Extract the model's response from the API response
        print(response)
        message = response.choices[0].message['content'].strip()
    
    else:
        # Use completions API for non-chat models
        response = openai.Completion.create(
            model=model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            top_p=top_p,
            logprobs=5,
            temperature=temperature
        )
        # Extract the model's response from the API response
        print(response)
        message = response.choices[0].text.strip()
    
    return message, response
