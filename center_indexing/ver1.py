import os
import glob
import subprocess
import re
import tqdm
import tempfile
from skopt import gp_minimize
from skopt.space import Real

from extract_chunk_data import extract_chunk_data
from calculate_combined_metric import calculate_combined_metric

def create_temporary_geomfile(x, y, geom_template, out_geom_path):
    """
    Writes a geometry file to 'out_geom_path', updating p0/corner_x and p0/corner_y.
    """
    with open(geom_template, 'r') as infile, open(out_geom_path, 'w') as outfile:
        for line in infile:
            if line.startswith("p0/corner_x"):
                outfile.write(f"p0/corner_x = {x}\n")
            elif line.startswith("p0/corner_y"):
                outfile.write(f"p0/corner_y = {y}\n")
            else:
                outfile.write(line)
    return out_geom_path

def run_indexing(listfile_path, geom_file, cell_file_path, out_prefix, num_threads=23):
    """
    Runs indexamajig and returns the path to the .stream output.
    If something fails, returns None.
    """
    output_stream = f"{out_prefix}.stream"

    cmd_str = (
        "indexamajig "
        f"-g {geom_file} "
        f"-p {cell_file_path} "
        f"-i {listfile_path} "
        f"-o {output_stream} "
        f"-j {num_threads} "
        "--peaks=cxi "
        "--indexing=xgandalf "
        "--xgandalf-sampling-pitch=5 "
        "--xgandalf-grad-desc-iterations=1 "
        "--xgandalf-tolerance=0.02 "
        "--xgandalf-max-peaks=50 "
        "--no-half-pixel-shift "
        "--no-non-hits-in-stream "
        "--no-image-data "
        "--no-revalidate "
        # "--no-refine "
        # "--no-retry "
        "--tolerance=5,5,5,5 "
    )

    try:
        subprocess.run(cmd_str, shell=True, check=True)
        return output_stream
    except subprocess.CalledProcessError:
        return None

def check_if_indexed(stream_file):
    """
    Simple heuristic: return True if 'Cell parameters' is present in the .stream
    """
    if not os.path.exists(stream_file):
        return False
    with open(stream_file, "r") as sf:
        content = sf.read()
        return ("Cell parameters" in content)

def parse_stream_for_metrics_no_norm(stream_file_path, metric_weights=None):
    """
    Parses the .stream file WITHOUT normalizing metrics.
    Prints raw metric values and computes a combined metric 
    directly from unnormalized values.
    
    Returns:
      results: list of (filename, event_number, combined_metric, chunk_content), sorted ascending
      none_results: list of failed entries
      header: the top portion of the .stream file
    """
    metric_names = [
        'weighted_rmsd', 'length_deviation', 'angle_deviation', 'num_peaks',
        'num_reflections', 'peak_resolution', 'diffraction_resolution', 'profile_radius',
        'percentage_indexed'
    ]

    # Default or user-provided weights
    if metric_weights is None:
        metric_weights = {
            'weighted_rmsd': 0.1,
            'length_deviation': 1,
            'angle_deviation': 1,
            'num_peaks': -0.001,
            'num_reflections': 0.0001,
            'peak_resolution': -0.1,
            'diffraction_resolution': 0.1,
            'profile_radius': 100,
            'percentage_indexed': -0.01
        }
    elif isinstance(metric_weights, (list, tuple)) and len(metric_weights) == len(metric_names):
        metric_weights = dict(zip(metric_names, metric_weights))

    results = []
    none_results = []

    with open(stream_file_path, 'r') as f:
        content = f.read()
        header, *chunks = re.split(r'----- Begin chunk -----', content)

        # If needed, read original cell params from header
        cell_params_match = re.search(
            r'a = ([\d.]+) A\nb = ([\d.]+) A\nc = ([\d.]+) A\n'
            r'al = ([\d.]+) deg\nbe = ([\d.]+) deg\nga = ([\d.]+) deg',
            header
        )
        original_cell_params = tuple(map(float, cell_params_match.groups())) if cell_params_match else None

        for chunk in tqdm.tqdm(chunks, desc=f"Parsing chunks in {os.path.basename(stream_file_path)}", unit="chunk"):
            if "indexed_by = none" in chunk.lower():
                # Skip unindexed
                continue

            extracted = extract_chunk_data(chunk, original_cell_params)
            (
                event_number, weighted_rmsd, length_deviation, angle_deviation,
                num_peaks, num_reflections, peak_resolution, diffraction_resolution,
                profile_radius, percentage_indexed, chunk_content
            ) = extracted

            if event_number is not None and None not in (
                weighted_rmsd, length_deviation, angle_deviation,
                peak_resolution, diffraction_resolution, percentage_indexed
            ):

                print(f"\n--- Chunk (event {event_number}) Metrics ---")
                print(f"  weighted_rmsd        = {weighted_rmsd}    weighted value = {weighted_rmsd}*{metric_weights['weighted_rmsd']} = {weighted_rmsd * metric_weights['weighted_rmsd']}")
                print(f"  length_deviation     = {length_deviation}    weighted value = {length_deviation}*{metric_weights['length_deviation']} = {length_deviation * metric_weights['length_deviation']}")
                print(f"  angle_deviation      = {angle_deviation}    weighted value = {angle_deviation}*{metric_weights['angle_deviation']} = {angle_deviation * metric_weights['angle_deviation']}")
                print(f"  num_peaks            = {num_peaks}    weighted value = {num_peaks}*{metric_weights['num_peaks']} = {num_peaks * metric_weights['num_peaks']}")
                print(f"  num_reflections      = {num_reflections}    weighted value = {num_reflections}*{metric_weights['num_reflections']} = {num_reflections * metric_weights['num_reflections']}")
                print(f"  peak_resolution      = {peak_resolution}    weighted value = {peak_resolution}*{metric_weights['peak_resolution']} = {peak_resolution * metric_weights['peak_resolution']}")
                print(f"  diffraction_resolution = {diffraction_resolution}    weighted value = {diffraction_resolution}*{metric_weights['diffraction_resolution']} = {diffraction_resolution * metric_weights['diffraction_resolution']}")
                print(f"  profile_radius       = {profile_radius}    weighted value = {profile_radius}*{metric_weights['profile_radius']} = {profile_radius * metric_weights['profile_radius']}")
                print(f"  percentage_indexed   = {percentage_indexed}    weighted value = {percentage_indexed}*{metric_weights['percentage_indexed']} = {percentage_indexed * metric_weights['percentage_indexed']}")


                # Compute combined metric directly from raw values
                # We'll assemble them into an 'all_metrics' dict for a single chunk
                # so that your 'calculate_combined_metric()' can still be used.
                single_chunk_metrics = {
                    'weighted_rmsd': [weighted_rmsd],
                    'length_deviation': [length_deviation],
                    'angle_deviation': [angle_deviation],
                    'num_peaks': [num_peaks],
                    'num_reflections': [num_reflections],
                    'peak_resolution': [peak_resolution],
                    'diffraction_resolution': [diffraction_resolution],
                    'profile_radius': [profile_radius],
                    'percentage_indexed': [percentage_indexed]
                }

                # In 'calculate_combined_metric', the first argument is an index i.
                # We'll just use i=0 for a single-chunk approach. 
                combined_metric = calculate_combined_metric(0, single_chunk_metrics, metric_weights)
                
                results.append((
                    os.path.basename(stream_file_path),
                    event_number,
                    combined_metric,
                    chunk_content
                ))
            else:
                none_results.append((os.path.basename(stream_file_path), event_number, "None"))

    # Sort ascending by combined_metric
    results.sort(key=lambda x: x[2])
    return results, none_results, header

def compute_iqm_single_value_no_norm(stream_file_path, metric_weights=None):
    """
    Return the best (lowest) combined metric from parse_stream_for_metrics_no_norm
    or 1e6 if none is valid.
    """
    results, none_results, _header = parse_stream_for_metrics_no_norm(stream_file_path, metric_weights)
    if not results or results[0][2] >= 1e6:
        return 1e6
    best_val = results[0][2]
    print(f"Best combined metric from {os.path.basename(stream_file_path)} = {best_val}")
    return best_val

def objective(center, h5_list_file, geom_template, cell_file_path, working_dir):
    """
    The objective to minimize: runs indexing with the given (x, y),
    uses raw metric values (no normalization), prints them,
    and returns the numeric combined metric. Keeps the .stream, removes only the geom.
    """
    x, y = center
    
    # 1) Create a temporary geometry file
    with tempfile.NamedTemporaryFile(suffix=".geom", delete=False) as tmp_geom:
        geom_path = tmp_geom.name
        create_temporary_geomfile(x, y, geom_template, geom_path)

    # 2) Build out_prefix for the .stream
    out_prefix = os.path.join(working_dir, f"index_{x:.2f}_{y:.2f}")
    stream_file = run_indexing(h5_list_file, geom_path, cell_file_path, out_prefix)

    # Remove the geometry file immediately after usage
    if os.path.exists(geom_path):
        os.remove(geom_path)

    # If indexing fails => big penalty
    if stream_file is None:
        return 1e6

    # Check if anything indexed => if not, penalty
    if not check_if_indexed(stream_file):
        return 1e6

    # 3) Compute IQM from raw (unnormalized) metrics
    iqm_value = compute_iqm_single_value_no_norm(stream_file)

    # Keep the .stream file (do NOT remove it)
    return iqm_value

def optimize_center_for_frame(h5_file, event_number, geom_template, cell_file_path, working_dir, n_calls=20):
    """
    Performs a Bayesian optimization over (x, y) corners for a single frame,
    using raw metric values (no normalization).
    """

    # Specify the directory where your .stream files are located
    folder_path = working_dir  # Update this path accordingly

    # Use glob to find all files ending with .stream in the specified folder
    stream_files = glob.glob(os.path.join(folder_path, '*.stream'))

    # Iterate over the list of .stream files and delete each one
    for filepath in stream_files:
        try:
            os.remove(filepath)
            print(f"Deleted {filepath}")
        except Exception as e:
            print(f"Error deleting {filepath}: {e}")

    # 1) .list file for a single HDF5 frame (//64 if you need event #64)
    h5_list_file = os.path.join(working_dir, "single_frame.list")
    with open(h5_list_file, "w") as f_list:
        f_list.write(f"{h5_file} //{event_number}\n")

    # 2) Define search space
    space = [
        Real(-514, -510, name='x'),
        Real(-514, -510, name='y')
    ]

    # 3) Wrap objective
    def wrapped_objective(params):
        return objective(params, h5_list_file, geom_template, cell_file_path, working_dir)

    # 4) Run Bayesian optimization
    result = gp_minimize(
        wrapped_objective,
        space,
        n_calls=n_calls,
        random_state=42
    )

    best_x, best_y = result.x
    best_iqm = result.fun
    print(f"Best center found: (x={best_x:.2f}, y={best_y:.2f}), IQM={best_iqm:.4f}")
    return (best_x, best_y, best_iqm)

# -----------------
# Example usage
# -----------------
if __name__ == "__main__":
    single_h5_frame   = "/home/bubl3932/files/UOX1/UOXs_FCI-1/UOXs.h5"
    event_number = 4
    cell_file_path    = "/home/bubl3932/files/UOX1/UOX.cell"
    base_geom_template= "/home/bubl3932/files/UOX1/UOX.geom"
    tmp_work_dir      = "/home/bubl3932/files/UOX1/UOXs_FCI-1"

    best_center = optimize_center_for_frame(
        h5_file=single_h5_frame,
        event_number=event_number,
        cell_file_path=cell_file_path,
        geom_template=base_geom_template,
        working_dir=tmp_work_dir,
        n_calls=30
    )
    print("Done. Best center:", best_center)
