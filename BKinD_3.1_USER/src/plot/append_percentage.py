# append_percentage.py

# Standard Library Imports
import re

# # Function to append '%' to any decimal number
# def append_percentage(source):
#     return re.sub(r'(\.\d)', r'\1%', source)

def append_percentage(source):
    # Drop '.0' and add '%' to decimals
    source = re.sub(r'\.0\b', '%', source)
    # Add '%' to any remaining decimals
    source = re.sub(r'(\.\d)', r'\1%', source)
    return source
