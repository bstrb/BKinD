import os
import subprocess
import re
import tqdm
import tempfile
from skopt import gp_minimize
from skopt.space import Real  # or Integer, depending on your usage

from extract_chunk_data import extract_chunk_data
from calculate_combined_metric import calculate_combined_metric

def create_temporary_geomfile(x, y, geom_template, out_geom_path):
    """
    Create a temporary geometry file at `out_geom_path`, 
    with modified x, y corner values in the geometry.
    """
    with open(geom_template, 'r') as infile, open(out_geom_path, 'w') as outfile:
        lines = infile.readlines()
    
        for line in lines:
            if line.startswith("p0/corner_x"):
                outfile.write(f"p0/corner_x = {x}\n")
            elif line.startswith("p0/corner_y"):
                outfile.write(f"p0/corner_y = {y}\n")
            else:
                outfile.write(line)

    return out_geom_path  # Return the path to the geom file we just wrote

def run_indexing(listfile_path, geom_file, cell_file_path, out_prefix, num_threads=23):
    """
    Run indexamajig (or another indexing tool) and return the path
    to its .stream output. 
    Assumes indexamajig is in the PATH or specify the full path in the cmd.
    """
    output_stream = f"{out_prefix}.stream"
    
    # Example indexamajig command; adapt as necessary.
    # Note: Using a single string with shell=True is another approach,
    # but here we show a list so we can pass parameters more cleanly.
    cmd = [
        "indexamajig "
        f"-g {geom_file} "
        f"-p {cell_file_path} "
        f"-i {listfile_path} "
        f"-o {output_stream} "
        f"-j {num_threads} "
        "--indexing=xgandalf "
        "--xgandalf-sampling-pitch=5 "
        "--xgandalf-grad-desc-iterations=1 "
        "--xgandalf-tolerance=0.02 "
        "--xgandalf-max-peaks=50 "
        "--no-half-pixel-shift "
        "--peaks=cxi "
        "--no-non-hits-in-stream "
        "--no-image-data "
        "--no-refine "
        "--no-revalidate "
        "--no-retry "
        # You can set a tolerance with multiple values by repeating or as a single string:
        "--tolerance=5,5,5,5 "
    ]
    
    try:
        subprocess.run(cmd, shell=True, check=True)
        return output_stream
    except subprocess.CalledProcessError:
        # If something fails, return None (penalize in the objective)
        return None


def check_if_indexed(stream_file):
    """
    Check the stream file to see if indexing was successful.
    Return True if at least one crystal was found, otherwise False.
    """
    if not os.path.exists(stream_file):
        return False
    
    with open(stream_file, "r") as sf:
        content = sf.read()
        # A simple heuristic: look for 'Cell parameters' 
        # which usually indicates at least one indexed crystal.
        if "Cell parameters" in content:
            return True
    
    return False


def parse_stream_for_metrics(stream_file_path, metric_weights=None):
    """
    Parse the .stream file to extract metrics, compute a combined metric for each chunk,
    and return a sorted list of (filename, event_number, combined_metric, chunk_content),
    plus unparseable entries (none_results) and the stream header.
    
    This is your old `compute_iqm` logic, but renamed for clarity.
    """
    metric_names = [
        'weighted_rmsd', 'length_deviation', 'angle_deviation', 'num_peaks',
        'num_reflections', 'peak_resolution', 'diffraction_resolution', 'profile_radius',
        'percentage_indexed'  # New metric
    ]

    # If metric_weights is a list or tuple, map it to a dict
    if isinstance(metric_weights, (list, tuple)) and len(metric_weights) == len(metric_names):
        metric_weights = dict(zip(metric_names, metric_weights))
    elif metric_weights is None:
        # Default weights
        metric_weights = {
            'weighted_rmsd': 12,
            'length_deviation': 12,
            'angle_deviation': 10,
            'num_peaks': -12,
            'num_reflections': 12,
            'peak_resolution': -15,
            'diffraction_resolution': 10,
            'profile_radius': 13,
            'percentage_indexed': -13  # Negative to favor higher % indexed
        }

    results = []
    none_results = []
    all_metrics = {k: [] for k in metric_names}  # Collect metrics across chunks

    with open(stream_file_path, 'r') as f:
        content = f.read()
        header, *chunks = re.split(r'----- Begin chunk -----', content)

        # Extract original cell parameters from the header if needed
        cell_params_match = re.search(
            r'a = ([\d.]+) A\nb = ([\d.]+) A\nc = ([\d.]+) A\n'
            r'al = ([\d.]+) deg\nbe = ([\d.]+) deg\nga = ([\d.]+) deg',
            header
        )
        if cell_params_match:
            original_cell_params = tuple(map(float, cell_params_match.groups()))
        else:
            original_cell_params = None

        # Iterate over each chunk
        for chunk in tqdm.tqdm(chunks, 
                               desc=f"Processing chunks in {os.path.basename(stream_file_path)}",
                               unit="chunk"):
            if "indexed_by = none" in chunk.lower():
                continue  # Skip unindexed chunks

            (
                event_number, weighted_rmsd, length_deviation, angle_deviation,
                num_peaks, num_reflections, peak_resolution, diffraction_resolution,
                profile_radius, percentage_indexed, chunk_content
            ) = extract_chunk_data(chunk, original_cell_params)

            # If extraction succeeded and no metric is None
            if event_number is not None and None not in (
                weighted_rmsd, length_deviation, angle_deviation, 
                peak_resolution, diffraction_resolution, percentage_indexed
            ):
                results.append((
                    os.path.basename(stream_file_path),
                    event_number,
                    weighted_rmsd,
                    length_deviation,
                    angle_deviation,
                    num_peaks,
                    num_reflections,
                    peak_resolution,
                    diffraction_resolution,
                    profile_radius,
                    percentage_indexed,
                    chunk_content
                ))

                all_metrics['weighted_rmsd'].append(weighted_rmsd)
                all_metrics['length_deviation'].append(length_deviation)
                all_metrics['angle_deviation'].append(angle_deviation)
                all_metrics['num_peaks'].append(num_peaks)
                all_metrics['num_reflections'].append(num_reflections)
                all_metrics['peak_resolution'].append(peak_resolution)
                all_metrics['diffraction_resolution'].append(diffraction_resolution)
                all_metrics['profile_radius'].append(profile_radius)
                all_metrics['percentage_indexed'].append(percentage_indexed)

            else:
                none_results.append((os.path.basename(stream_file_path), event_number, "None"))

    # Normalize each metric to [0..1] across all chunks
    for key, values in all_metrics.items():
        if values:  # If not empty
            min_val, max_val = min(values), max(values)
            if max_val != min_val:
                normed = [(v - min_val) / (max_val - min_val) for v in values]
            else:
                # If all values are the same, assign them all to 0.5 (or 1, your choice)
                normed = [0.5 for _ in values]
            all_metrics[key] = normed
        else:
            # No data for this metric
            all_metrics[key] = []

    # Update results with normalized metrics, compute combined metric
    for i, row in enumerate(results):
        (
            fname, event_number,
            weighted_rmsd, length_deviation, angle_deviation,
            num_peaks, num_reflections, peak_resolution,
            diffraction_resolution, profile_radius, percentage_indexed,
            chunk_content
        ) = row

        combined_metric = calculate_combined_metric(i, all_metrics, metric_weights)
        # Overwrite row with combined_metric
        results[i] = (fname, event_number, combined_metric, chunk_content)

    # Sort by combined_metric ascending
    results.sort(key=lambda x: x[2])  # x[2] is the combined metric

    return results, none_results, header


def compute_iqm_single_value(stream_file_path, metric_weights=None):
    """
    Convenience wrapper that parses the .stream file, 
    returns a single numeric IQM (the lowest combined metric across chunks).
    """
    results, none_results, header = parse_stream_for_metrics(stream_file_path, metric_weights)
    if not results:
        # If no valid indexed chunks, return a large penalty
        return 1e6
    # results is sorted ascending by combined metric, so take the first
    best_combined_metric = results[0][2]
    print(best_combined_metric)
    return best_combined_metric


def objective(center, h5_list_file, geom_template, cell_file_path, working_dir):
    """
    Objective function to be minimized by skopt.
    center: (x, y)
    h5_list_file: .list file referencing the single .h5 event
    geom_template: base geometry template
    working_dir: folder for temp outputs
    """
    x, y = center
    
    # 1. Create a temporary geomfile
    temp_geom_path = os.path.join(working_dir, f"temp_geom_{x:.2f}_{y:.2f}.geom")
    create_temporary_geomfile(x, y, geom_template, temp_geom_path)
    
    # 2. Run indexamajig
    out_prefix = os.path.join(working_dir, f"index_{x:.2f}_{y:.2f}")
    stream_file = run_indexing(h5_list_file, temp_geom_path, cell_file_path, out_prefix)
    
    if stream_file is None:
        # If indexing completely fails, return a large penalty
        return 1e6
    
    # 3. Check if anything was indexed
    if not check_if_indexed(stream_file):
        # Non-indexed => big penalty
        return 1e6
    
    # 4. Compute (and return) the single best IQM value from the .stream
    iqm_value = compute_iqm_single_value(stream_file)
    
    # Optional cleanup of files if desired
    # os.remove(temp_geom_path)
    # os.remove(stream_file)
    
    return iqm_value  # Minimizing this


def optimize_center_for_frame(h5_file, geom_template, cell_file_path, working_dir, n_calls=20):
    """
    Optimize (x, y) for a single frame:
      - h5_file: path to a single-frame HDF5
      - geom_template: path to base geometry
      - working_dir: temp files output
      - n_calls: number of skopt evaluations (Bayesian steps)
    Returns: (best_x, best_y, best_iqm)
    """
    # 1. Create a .list file referencing this single .h5
    h5_list_file = os.path.join(working_dir, "single_frame.list")
    with open(h5_list_file, "w") as f_list:
        f_list.write(f"{h5_file} //64\n")
    
    # 2. Define the search space for (x, y) 
    # Adjust as needed to your detector geometry or initial guess range
    space = [
        Real(-514, -510, name='x'),
        Real(-514, -510, name='y')
    ]
    
    # 3. Wrap the objective to inject fixed parameters
    def wrapped_objective(params):
        return objective(params, h5_list_file, geom_template, cell_file_path, working_dir)
    
    # 4. Perform Bayesian optimization (with Gaussian Process surrogate)
    res = gp_minimize(
        wrapped_objective,
        space,
        n_calls=n_calls,
        random_state=42
    )
    
    best_x, best_y = res.x
    best_iqm = res.fun
    print(f"Best center found: (x={best_x:.2f}, y={best_y:.2f}) with IQM={best_iqm:.4f}")
    
    return (best_x, best_y, best_iqm)


# ----------------------
# Example usage
# ----------------------
if __name__ == "__main__":
    # Example placeholders (adapt to your environment)
    single_h5_frame = "/home/bubl3932/files/UOX1/UOXs_FCI-1/UOXs.h5"
    cell_file_path = "/home/bubl3932/files/UOX1/UOX.cell"
    base_geom_template = "/home/bubl3932/files/UOX1/UOX.geom"
    tmp_work_dir = "/home/bubl3932/files/UOX1/UOXs_FCI-1"
    
    # Optimize center
    best_center = optimize_center_for_frame(
        h5_file=single_h5_frame,
        cell_file_path=cell_file_path,
        geom_template=base_geom_template,
        working_dir=tmp_work_dir,
        n_calls=30
    )
    print("Done. Best center:", best_center)
