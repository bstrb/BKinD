#!/usr/bin/env python3

import sys
import os

def count_sol_lines(input_file):

    # Split the input filename into base and extension
    base, ext = os.path.splitext(input_file)

    # Construct the corresponding .sol filename
    sol_file = base + ".sol"

    # Check if the .sol file exists
    if not os.path.isfile(sol_file):
        print(f"Error: File '{sol_file}' not found.")
        sys.exit(1)

    # Count the number of lines in the .sol file
    try:
        with open(sol_file, 'r') as file:
            line_count = sum(1 for _ in file)
        return line_count
    except Exception as e:
        print(f"Error reading file '{sol_file}': {e}")
        sys.exit(1)

