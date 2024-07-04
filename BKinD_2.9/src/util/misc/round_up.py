# round_up.py

# Standard Library Imports
import math

# Function to round to the closest decimal above
def round_up(value):
    return math.ceil(value * 10) / 10

# # Function to round to the closest part of 100 above
# def round_up(value):
#     return math.ceil(value * 100) / 100