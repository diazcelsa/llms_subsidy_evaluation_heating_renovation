import numpy as np
import re

def parse_survey_responses(text):
    # Extract responses using regular expressions
    responses = re.findall(r'(\d+)\.\s*([AaBbCc])\S*\)', text)
    
    # Check if there are exactly 15 responses
    if len(responses) != 15:
        return "Number of responses do not match 15"
    
    cleaned_responses = []
    
    # Clean and extract the relevant information (A/a, B/b, or C/c)
    for _, response in responses:
        cleaned_response = response.upper()
        if cleaned_response in {'A', 'B', 'C'}:
            cleaned_responses.append(cleaned_response)
    
    # Check if the responses are valid
    if len(cleaned_responses) != 15:
        return "Answer provided out of options"
    
    # Create a numpy array with the cleaned responses
    response_array = np.array(cleaned_responses)

    # Check if the order of responses is correct (1, 2, 3, ..., 15)
    expected_order = list(map(str, range(1, 16)))
    actual_order = [num for num, _ in responses]
    if not np.array_equal(actual_order, expected_order):
        return "Order of answers incorrect.\nExpected order: {expected_order}\nActual order: {actual_order}"
    
    return response_array