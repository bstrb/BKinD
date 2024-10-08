# plot_chunk_merge_def.py

from process_and_plot_all_files import process_and_plot_all_files
from process_and_plot_final_rfree import process_and_plot_final_rfree

def plot_chunk_merge(base_path, fit_exponential=False):

    process_and_plot_all_files(base_path)

    process_and_plot_final_rfree(base_path, fit_exponential)