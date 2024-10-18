# rcis_main.py
from find_first_file import find_first_file
from automate_evaluation_and_integration import automate_evaluation_and_integration

def main():
    stream_file_folder = "/home/buster/UOXm/5x5_0-01"
    cellfile_path = find_first_file(stream_file_folder, ".cell")
    pdb_file = "/home/buster/UOXm/5x5_0-01/UOX.pdb"
    weights_list = [(1,0,0),(1,1,0),(1,4,0)]
    lattice = "oI"  # Pearson Symbol read from .cell file first letter for lattice type and second for centering
    ring_size = [(3, 4, 7)]
    pointgroup = "mmm"
    num_threads = 23
    bins = 20
    iterations = 3

    automate_evaluation_and_integration(stream_file_folder, weights_list, lattice, ring_size, cellfile_path, pointgroup, num_threads, bins, pdb_file, iterations)

if __name__ == "__main__":
    main()
