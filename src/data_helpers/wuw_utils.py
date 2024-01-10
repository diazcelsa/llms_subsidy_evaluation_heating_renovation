import re
import pandas as pd


def standardize_section_headers(file_path, encoding='ISO-8859-1'):
    """
    Standardize the section headers in the text file to make them easily identifiable and consistent.

    Args:
    file_path (str): The path to the original text file.

    Returns:
    str: A string representing the text with standardized section headers.
    """
    with open(file_path, 'r', encoding=encoding) as file:
        lines = file.readlines()

    standardized_lines = []
    header_detected = False

    for line in lines:
        if '----' in line:
            # Detecting the start and end of a section header
            header_detected = not header_detected
            if header_detected:
                # Start of a new section header, add a marker
                standardized_lines.append("\n---NEW SECTION START---\n")
            else:
                # End of a section header, add a marker
                standardized_lines.append("\n---SECTION CONTENT---\n")
        else:
            if header_detected:
                # Process the header line
                standardized_lines.append(line.strip() + ' ')
            else:
                # Regular line, add it as is
                standardized_lines.append(line)

    return ''.join(standardized_lines)

def normalize_key_value_pairs(file_path, encoding='ISO-8859-1'):
    """
    Normalize the key-value pairs in the text file to ensure they follow a consistent and easily parsable format.

    Args:
    file_path (str): The path to the text file with standardized section headers.

    Returns:
    str: A string representing the text with normalized key-value pairs.
    """
    with open(file_path, 'r', encoding=encoding) as file:
        lines = file.readlines()

    normalized_lines = []
    in_content_section = False

    for line in lines:
        if '---NEW SECTION START---' in line:
            in_content_section = False
            normalized_lines.append(line)
        elif '---SECTION CONTENT---' in line:
            in_content_section = True
            normalized_lines.append(line)
        else:
            if in_content_section:
                # Process the key-value pairs
                if ':' in line:
                    # Split the line into key and value
                    key, value = line.split(':', 1)
                    normalized_lines.append(key.strip() + ': ' + value.strip() + '\n')
                else:
                    # Continuation of a value, append it to the last key-value pair
                    if normalized_lines and ':' in normalized_lines[-1]:
                        normalized_lines[-1] = normalized_lines[-1].strip() + ' ' + line.strip() + '\n'
            else:
                # Regular line, add it as is
                normalized_lines.append(line)

    return ''.join(normalized_lines)

def update_reformatting(file_path, encoding='ISO-8859-1'):
    """
    Update the reformatting of the text file to correctly handle section headers and key-value pairs:
    1) Only move left side content to a new line between ---NEW SECTION START--- and ---SECTION CONTENT--- 
       if there is content on both the left and right sides.
    2) Keep key-value pairs on the same line between ---SECTION CONTENT--- and ---NEW SECTION START---, 
       but move new key-value pairs to a new line if they are found after a large space.

    Args:
    file_path (str): The path to the previously normalized text file.

    Returns:
    str: A string representing the correctly reformatted text.
    """
    with open(file_path, 'r', encoding=encoding) as file:
        lines = file.readlines()

    updated_lines = []
    in_section_header = False
    in_content_section = False

    for line in lines:
        if '---NEW SECTION START---' in line:
            in_section_header = True
            in_content_section = False
            updated_lines.append(line)
        elif '---SECTION CONTENT---' in line:
            in_section_header = False
            in_content_section = True
            updated_lines.append(line)
        else:
            if in_section_header:
                # Check if there is content on both left and right sides
                if ' ' in line.strip() and not line.strip().endswith(':'):
                    left, right = line.rsplit(' ', 1)
                    if right.strip():
                        # Add left side content to a new line only if right side content is present
                        updated_lines.append(left.strip() + '\n')
                        updated_lines.append(right.strip() + '\n')
                    else:
                        updated_lines.append(line)
                else:
                    updated_lines.append(line)
            elif in_content_section:
                # Keep key-value pairs on the same line
                # Move new key-value pairs to a new line if found after a large space
                split_lines = re.split(r'\s{2,}(?=\w+:)', line)
                updated_lines.extend([s.strip() + '\n' for s in split_lines])
            else:
                updated_lines.append(line)

    return ''.join(updated_lines)

def final_reformatting(file_path, encoding='ISO-8859-1'):
    """
    Final reformatting of the text file to:
    1) Move 'Missing' key-value pairs to a new line if found after 'Unique values'.
    2) In section headers, create a new line with the second string if different from the first one.

    Args:
    file_path (str): The path to the previously reformatted text file.

    Returns:
    str: A string representing the text with final reformatting applied.
    """
    with open(file_path, 'r', encoding=encoding) as file:
        lines = file.readlines()

    final_lines = []
    in_section_header = False

    for line in lines:
        if '---NEW SECTION START---' in line:
            in_section_header = True
            final_lines.append(line)
        elif '---SECTION CONTENT---' in line:
            in_section_header = False
            final_lines.append(line)
        else:
            if in_section_header:
                # Split the line if it contains two different strings separated by a large space
                parts = line.split()
                if len(parts) > 1 and parts[0] != parts[-1]:
                    final_lines.append(parts[0] + '\n')
                    final_lines.append(' '.join(parts[1:]) + '\n')
                else:
                    final_lines.append(line)
            else:
                # Move 'Missing' key-value pairs to a new line
                split_lines = re.split(r'\s{2,}(?=\bMissing\b:)', line)
                final_lines.extend([s.strip() + '\n' for s in split_lines])

    return ''.join(final_lines)

def remove_duplicate_headers_corrected(file_path, encoding='ISO-8859-1'):
    """
    Corrected function to remove duplicate headers in the section headers of the text file.

    Args:
    file_path (str): The path to the text file.

    Returns:
    str: A string representing the text with duplicate headers removed.
    """
    with open(file_path, 'r', encoding=encoding) as file:
        lines = file.readlines()

    updated_lines = []
    for line in lines:
        if '---NEW SECTION START---' in line:
            updated_lines.append(line)
            continue

        # Check for duplicate headers
        parts = line.strip().split()
        if len(parts) > 1 and parts[0] == parts[-1]:
            updated_lines.append(parts[0] + '\n')
        else:
            updated_lines.append(line)

    return ''.join(updated_lines)

def create_dataframe_from_text(file_path, encoding='ISO-8859-1'):
    """
    Revised function to create a DataFrame from the text file, splitting 'name' and 'meaning' correctly
    based on the presence of a second line in the section header.

    Args:
    file_path (str): The path to the text file.

    Returns:
    pandas.DataFrame: A DataFrame created from the text file.
    """
    with open(file_path, 'r', encoding=encoding) as file:
        lines = file.readlines()

    section_data = []
    current_section = {}
    in_section_header = False
    in_content_section = False

    for line in lines:
        if '---NEW SECTION START---' in line:
            in_section_header = True
            in_content_section = False
            # If a section was already being processed, add it to the list
            if current_section:
                section_data.append(current_section)
                current_section = {}
        elif '---SECTION CONTENT---' in line:
            in_section_header = False
            in_content_section = True
        else:
            if in_section_header:
                # Process section header
                if 'name' not in current_section:
                    current_section['name'] = line.strip()
                else:
                    current_section['meaning'] = line.strip()
            elif in_content_section:
                # Process section content
                if ':' in line:
                    key, value = line.split(':', 1)
                    current_section[key.strip()] = value.strip()

    # Add the last section if it exists
    if current_section:
        section_data.append(current_section)

    return pd.DataFrame(section_data)

def split_unique_values_and_missing(df):
    """
    Split the 'Unique values' column by the last character before 'Missing' and add what comes after 
    'Missing' in a new column 'Missing'.

    Args:
    df (pandas.DataFrame): The DataFrame to process.

    Returns:
    pandas.DataFrame: The processed DataFrame with the 'Missing' column added.
    """
    # Copy the DataFrame to avoid modifying the original
    df_processed = df.copy()

    # Splitting 'Unique values' column and adding 'Missing' column
    df_processed[['Unique values', 'Missing']] = df_processed['Unique values'].str.split('Missing', expand=True)
    df_processed['Unique values'] = df_processed['Unique values'].str.rstrip()  # Remove trailing spaces
    df_processed['Missing'] = df_processed['Missing'].str.lstrip(': ').replace('.', '')  # Format 'Missing' column

    return df_processed