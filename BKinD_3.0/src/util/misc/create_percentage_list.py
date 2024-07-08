# create_percentage_list.py

# Utility Misc Imports
from util.misc.round_up import round_up

def create_percentage_list(start, end, step_size, num_steps, step_mode, include_steps=False):

    # start = round(start, 1)
    start = round_up(start)

    if include_steps:
        percentage_list = []

        if step_mode == "size":

            current = start - (start % step_size)  # Find the closest step_size value below the start percentage

            while current >= end:
                percentage_list.append(round(current, 1))
                current -= step_size

            if percentage_list[-1] != end:
                percentage_list.append(end)
            
            if not percentage_list[0] == start:
                # Add the start percentage at the beginning
                percentage_list.insert(0, start)
        else:

            # Calculate the step size
            step_size = round((end - start) / (num_steps + 1),1)

            # Generate the list
            percentage_list = [round(start + i * step_size,1) for i in range(num_steps + 2)]
    else:
        percentage_list = [start, end]

    return percentage_list
