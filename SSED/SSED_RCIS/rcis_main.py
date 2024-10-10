# rcis_main.py
from generate_bash_script import find_first_file
from automate_evaluation_and_integration import automate_evaluation_and_integration

def main():
    stream_file_folder = "/home/buster/UOX1/5x5"
    cellfile_path = find_first_file(stream_file_folder, ".cell")
    pdb_file = "/home/buster/UOX1/5x5/UOX.pdb"
    weights_list = [
    (1, 1, -1), (1, 1, 0), (1, 0, -1),
    (1, 0, 0), (0, 1, -1), (0, 1, 0),
    (0, 0, -1)
    ]
    lattice = "oI"  # Pearson Symbol read from .cell file first letter for lattice type and second for centering
    ring_size = [(3, 4, 7)]
    pointgroup = "mmm"
    num_threads = 23
    bins = 20

    automate_evaluation_and_integration(stream_file_folder, weights_list, lattice, ring_size, cellfile_path, pointgroup, num_threads, bins, pdb_file)

if __name__ == "__main__":
    main()
