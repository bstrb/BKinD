# create_percentage_list.py

# Utility Misc Imports
from util.misc.round_up import round_up

def create_percentage_list(start, end, step):

    # start = round(start, 1)
    start = round_up(start)
    
    if step <= 0:
        raise ValueError("Step size must be positive")

    percentage_list = []
    current = start - (start % step)  # Find the closest step value below the start percentage

    while current >= end:
        percentage_list.append(round(current, 1))
        current -= step

    if percentage_list[-1] != end:
        percentage_list.append(end)
    
    if not percentage_list[0] == start:
        # Add the start percentage at the beginning
        percentage_list.insert(0, start)

    return percentage_list
