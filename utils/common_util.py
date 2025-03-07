"""
Copyright (c) 2025 by yuanzhenhui. All right reserved.
FilePath: /brain-mix/utils/common_util.py
Author: yuanzhenhui
Date: 2025-01-14 22:37:04
LastEditTime: 2025-01-14 22:43:56
"""

import os
import re
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check_variable(static_var, var):
    """
    Check if the variable is empty, and if so, return an empty string.
    Otherwise, return the variable with the static variable prepended.

    Parameters:
        static_var (str): The static variable to prepend if the variable is not empty.
        var (str): The variable to check.

    Returns:
        str: The variable with the static variable prepended if it is not empty, otherwise an empty string.
    """
    if var is None or len(str(var)) == 0:
        return ''
    else:
        return static_var+var
    
def format_number_with_leading_zero(number):
    """
    Format a number with a leading zero if it is a single digit.

    Parameters:
        number (int): The number to format.

    Returns:
        str: The formatted number as a string with a leading zero if it was a single digit.
    """
    number_as_string = str(number)
    if len(number_as_string) < 2:
        return "0" + number_as_string
    return number_as_string
    
def remove_folder(folder_path):
    """
    Recursively remove a folder and all its contents.

    This function deletes all the files and subdirectories within the specified
    folder, and then removes the folder itself.

    Parameters:
        folder_path (str): The path to the folder to be removed.

    Returns:
        None
    """
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Iterate over all items in the folder
        for item in os.listdir(folder_path):
            # Construct the full path to the item
            item_path = os.path.join(folder_path, item)
            # If the item is a file, remove it
            if os.path.isfile(item_path):
                os.remove(item_path)
            # If the item is a directory, recursively call remove_folder
            elif os.path.isdir(item_path):
                remove_folder(item_path)
        # Remove the now-empty folder
        os.rmdir(folder_path)


def split_array(lst, num):
    """
    Split a given list into a specified number of chunks.

    Parameters:
        lst (list): The list to split.
        num (int): The number of chunks to split the list into.

    Returns:
        list: A list of lists, where each sublist is a chunk of the original list.
    """
    if len(lst) < num:
        return [lst]  # If the list is shorter than the number of chunks, return the list as is

    # Calculate the average length of each chunk
    avg = len(lst) // num
    # Calculate the remainder of the division
    remainder = len(lst) % num

    # Initialize an empty list to store the chunks
    result = []
    # Initialize the start index to 0
    start = 0
    # Iterate over the number of chunks
    for i in range(num):
        # Calculate the end index for the current chunk
        end = start + avg + (1 if i < remainder else 0)
        # Append the current chunk to the result list
        result.append(lst[start:end])
        # Update the start index for the next chunk
        start = end

    # Return the list of chunks
    return result

def clean_markdown(text):
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'~~+', '', text)
    text = re.sub(r'$([^$]+)\]$[^$]+\)', r'\1', text)
    text = re.sub(r'!$([^$]*)\]$[^$]*\)', r'\1', text)
    text = re.sub(r'```\w*\n([\s\S]*?)\n```', r'\1', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'>\s?', '', text)
    text = re.sub(r'\|\s*(-+:|:-+|:-+:)\s*\|', '|', text)
    text = re.sub(r'<[^>]+>', '', text)
    return text
