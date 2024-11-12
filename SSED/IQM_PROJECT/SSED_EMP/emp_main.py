# rcis_main.py
from find_first_file import find_first_file
from automate_evaluation_and_integration import automate_evaluation_and_integration

def main():
    stream_file_folder = "/home/buster/UOX1/different_index_params/3x3_retry"
    cellfile_path = find_first_file(stream_file_folder, ".cell")
    pdb_file = find_first_file(stream_file_folder, ".pdb")
    weights_list = [(1, 2, 3, -1, 1, 1)] # w,l,a,p,r,g
    lattice = "oI"  # Pearson Symbol read from .cell file first letter for lattice type and second for centering
    ring_size = [(3, 4, 7)]
    pointgroup = "mmm"
    num_threads = 23
    bins = 20
    min_res = 1.5
    iterations = 3

    automate_evaluation_and_integration(stream_file_folder, weights_list, lattice, ring_size, cellfile_path, pointgroup, num_threads, bins, pdb_file, min_res, iterations)

if __name__ == "__main__":
    main()
