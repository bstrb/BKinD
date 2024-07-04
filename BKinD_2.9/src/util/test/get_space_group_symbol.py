# get_space_group_symbol.py

# %%

from cctbx import sgtbx

def get_space_group_symbol(space_group_number):
    space_group_info = sgtbx.space_group_info(number=space_group_number)
    symbol_with_number = space_group_info.symbol_and_number()
    symbol = symbol_with_number.split("(")[0].strip().replace(" ", "")
    # print(symbol)
    return symbol

# %%
