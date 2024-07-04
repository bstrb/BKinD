# get_space_group_symbol.py

# %%

from cctbx import sgtbx

def get_space_group_symbol(space_group_number):
    space_group_info = sgtbx.space_group_info(number=space_group_number)
    symbol_with_number = space_group_info.symbol_and_number()
    symbol = symbol_with_number #.split("(")[0].strip().replace(" ", "")
    return symbol

# Example usage
space_group_numbers = [221, 79, 19, 179]
for number in space_group_numbers:
    symbol = get_space_group_symbol(number)
    print(f'Space Group Number: {number}, Symbol: {symbol}')

# %%
