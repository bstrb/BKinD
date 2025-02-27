#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Refine the beam center (origin of reciprocal space) from a CrystFEL .stream file.
Optimized for large stream files by streaming line-by-line processing and reducing overhead.
"""

import sys
import argparse
import logging
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')  # For faster off-screen plotting if desired
import matplotlib.pyplot as plt

from find_beam_center import find_beam_center

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def parse_peaks(chunk_lines):
    """
    Parse peaks from chunk lines.
    Returns arrays for fs_px, ss_px, intensity.
    If no valid peaks found, returns None.
    """
    # Identify peaks section
    start_idx = None
    end_idx = None
    for i, line in enumerate(chunk_lines):
        if 'Peaks from peak search' in line:
            start_idx = i + 1
        elif 'End of peak list' in line:
            end_idx = i
            break
    
    if start_idx is None or end_idx is None:
        return None
    
    # The first line after 'Peaks from peak search' is the header
    header_line = chunk_lines[start_idx].strip().lower()
    headers = header_line.split()
    required = ['fs/px', 'ss/px', 'intensity']
    # Check if required columns exist
    if not all(r in headers for r in required):
        return None
    fs_idx = headers.index('fs/px')
    ss_idx = headers.index('ss/px')
    int_idx = headers.index('intensity')
    
    fs_list = []
    ss_list = []
    int_list = []
    for line in chunk_lines[start_idx+1:end_idx]:
        line = line.strip()
        if not line or line.lower().startswith('end'):
            continue
        parts = line.split()
        if len(parts) <= max(fs_idx, ss_idx, int_idx):
            continue
        try:
            fs_val = float(parts[fs_idx])
            ss_val = float(parts[ss_idx])
            i_val = float(parts[int_idx])
            fs_list.append(fs_val)
            ss_list.append(ss_val)
            int_list.append(i_val)
        except ValueError:
            continue
    
    if not fs_list:
        return None
    
    return np.array(fs_list), np.array(ss_list), np.array(int_list)

def parse_reflections(chunk_lines):
    """
    Parse reflections from chunk lines.
    Returns h, k, l, fs, ss, i as arrays, or None if not found.
    """
    start_idx = None
    end_idx = None
    for i, line in enumerate(chunk_lines):
        if 'Reflections measured after indexing' in line:
            start_idx = i + 1
        elif 'End of reflections' in line:
            end_idx = i
            break
    
    if start_idx is None or end_idx is None:
        return None
    
    # First line after start_idx is header
    header_line = chunk_lines[start_idx].strip().lower()
    headers = header_line.split()
    required = ['h', 'k', 'l', 'fs/px', 'ss/px', 'i']
    if not all(r in headers for r in required):
        return None
    
    h_idx = headers.index('h')
    k_idx = headers.index('k')
    l_idx = headers.index('l')
    fs_idx = headers.index('fs/px')
    ss_idx = headers.index('ss/px')
    i_idx = headers.index('i')
    
    h_list = []
    k_list = []
    l_list = []
    fs_list = []
    ss_list = []
    i_list = []
    
    for line in chunk_lines[start_idx+1:end_idx]:
        line = line.strip()
        if not line or line.lower().startswith('end'):
            continue
        parts = line.split()
        # Ensure we have enough columns
        if len(parts) <= max(h_idx, k_idx, l_idx, fs_idx, ss_idx, i_idx):
            continue
        try:
            h_val = float(parts[h_idx])
            k_val = float(parts[k_idx])
            l_val = float(parts[l_idx])
            fs_val = float(parts[fs_idx])
            ss_val = float(parts[ss_idx])
            i_val = float(parts[i_idx])
            h_list.append(h_val)
            k_list.append(k_val)
            l_list.append(l_val)
            fs_list.append(fs_val)
            ss_list.append(ss_val)
            i_list.append(i_val)
        except ValueError:
            continue
    
    if not h_list:
        return None
    
    return (np.array(h_list), np.array(k_list), np.array(l_list),
            np.array(fs_list), np.array(ss_list), np.array(i_list))

def match_peaks_reflections(fs_peaks, ss_peaks, fs_refl, ss_refl, h, k, l, tolerance=0.1, int_peaks=None, i_refl=None):
    """
    Matches peaks to reflections within a specified tolerance using a KD-tree.
    Returns a dictionary with matched coordinates and indices or None if no matches.
    """
    if fs_peaks.size == 0 or fs_refl.size == 0:
        return None
    
    ref_positions = np.column_stack((fs_refl, ss_refl))
    tree = cKDTree(ref_positions)
    
    peak_positions = np.column_stack((fs_peaks, ss_peaks))
    distances, indices = tree.query(peak_positions, distance_upper_bound=tolerance)
    
    valid_mask = (indices != len(ref_positions))
    if not np.any(valid_mask):
        return None
    
    matched_indices = indices[valid_mask]
    return {
        'fs_peak': fs_peaks[valid_mask],
        'ss_peak': ss_peaks[valid_mask],
        'int_peak': int_peaks[valid_mask] if int_peaks is not None else None,
        'fs_refl': fs_refl[matched_indices],
        'ss_refl': ss_refl[matched_indices],
        'h': h[matched_indices],
        'k': k[matched_indices],
        'l': l[matched_indices],
        'int_refl': i_refl[matched_indices] if i_refl is not None else None
    }

def process_stream(file_path, tolerance):
    """
    Process the .stream file line by line, extract beam centers.
    Returns a list of dictionaries with chunk_number, Xc_px, Yc_px.
    """
    beam_centers = []
    chunk_lines = []
    in_chunk = False
    chunk_count = 0
    
    with open(file_path, 'r') as f:
        for line in f:
            if '----- Begin chunk -----' in line:
                in_chunk = True
                chunk_lines = []
                continue
            if '----- End chunk -----' in line:
                # Process the chunk
                chunk_count += 1
                if chunk_count % 1000 == 0:
                    logger.info(f"Processed {chunk_count} chunks so far...")
                
                peaks = parse_peaks(chunk_lines)
                refl = parse_reflections(chunk_lines)
                
                if peaks is None or refl is None:
                    # Missing data
                    in_chunk = False
                    continue
                
                fs_peaks, ss_peaks, i_peaks = peaks
                h, k, l, fs_refl, ss_refl, i_refl = refl
                matched = match_peaks_reflections(fs_peaks, ss_peaks, fs_refl, ss_refl, h, k, l,
                                                  tolerance=tolerance, int_peaks=i_peaks, i_refl=i_refl)
                if matched is None or len(matched['fs_peak']) == 0:
                    # No matches
                    in_chunk = False
                    continue
                
                # Compute beam center
                x_data = matched['fs_refl']
                y_data = matched['ss_refl']
                h_data = matched['h']
                k_data = matched['k']
                l_data = matched['l']
                try:
                    Xc, Yc, params = find_beam_center(x_data, y_data, h_data, k_data, l_data)
                    beam_centers.append({'Chunk_Number': chunk_count, 'Xc_px': Xc, 'Yc_px': Yc})
                except Exception as e:
                    logger.error(f"Error computing beam center for chunk {chunk_count}: {e}")
                
                in_chunk = False
                continue
            
            if in_chunk:
                chunk_lines.append(line)
    
    return beam_centers
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description="Match peaks with reflections and find beam center per chunk."
    )

    # Set a default stream file path
    # default_stream = Path('/home/buster/UOX1/UOX1_bg_removed_new_PF/best_results_IQM_SUM_12_12_10_-12_12_-15_10_13_-13.stream')  # <-- Update this path as needed
    default_stream = Path('/home/buster/UOX123/3x3_retry/UOX_-512_-512.stream')  # <-- Update this path as needed
    
    # Make 'stream_file' optional by setting nargs='?' and providing a default
    parser.add_argument(
        'stream_file',
        type=Path,
        nargs='?',
        default=default_stream,
        help='Path to the .stream file. Defaults to path/to/default.stream if not provided.'
    )
    
    parser.add_argument(
        '--tolerance',
        type=float,
        default=0.5,
        help='Tolerance in pixels for matching peaks to reflections.'
    )
    
    parser.add_argument(
        '--output_csv',
        type=Path,
        help='Output CSV file for beam centers.'
    )
    
    parser.add_argument(
        '--output_plot_x',
        type=Path,
        help='Output plot image file for Xc vs Chunk_Number.'
    )
    
    parser.add_argument(
        '--output_plot_y',
        type=Path,
        help='Output plot image file for Yc vs Chunk_Number.'
    )
    
    args = parser.parse_args()
    
    # Determine the directory of the stream file
    stream_dir = args.stream_file.parent
    
    # Set default output paths if not provided
    args.output_csv = args.output_csv or stream_dir / 'beam_centers.csv'
    args.output_plot_x = args.output_plot_x or stream_dir / 'beam_centers_x_vs_chunk.png'
    args.output_plot_y = args.output_plot_y or stream_dir / 'beam_centers_y_vs_chunk.png'
    
    return args

def main():

    args = parse_args()

    if not args.stream_file.is_file():
        logger.error(f"The specified file does not exist: {args.stream_file}")
        sys.exit(1)

    logger.info("Starting stream processing...")
    beam_centers = process_stream(str(args.stream_file), args.tolerance)
    if not beam_centers:
        logger.warning("No beam centers were refined. Exiting.")
        sys.exit(0)

    # Create arrays for plotting and stats
    centers_array = np.array([(d['Chunk_Number'], d['Xc_px'], d['Yc_px']) for d in beam_centers])
    chunk_nums = centers_array[:,0]
    Xc_values = centers_array[:,1]
    Yc_values = centers_array[:,2]

    # Save to CSV
    import pandas as pd
    centers_df = pd.DataFrame(beam_centers)
    centers_df.to_csv(args.output_csv, index=False)
    logger.info(f"Beam centers saved to {args.output_csv}")

    # Compute median and simple ±2 range for plotting
    median_Xc = np.median(Xc_values)
    median_Yc = np.median(Yc_values)

    y_min_X = median_Xc - 2 
    y_max_X = median_Xc + 4 
    y_min_Y = median_Yc - 2 
    y_max_Y = median_Yc + 2 

    # Plot Xc_px vs Chunk_Number
    plt.figure(figsize=(10, 5))
    plt.scatter(chunk_nums, Xc_values, c='red', marker='o', label='Xc_px', s=10)
    plt.plot(chunk_nums, Xc_values, linestyle='-', color='red', linewidth=0.5)
    plt.axhline(median_Xc, color='green', linestyle='--', label='median Xc_px')
    plt.title('Beam Center X Coordinate vs Chunk Number')
    plt.xlabel('Chunk Number')
    plt.ylabel('X Coordinate (pixels)')
    plt.legend()
    plt.grid(True)
    plt.ylim(y_min_X, y_max_X)
    plt.tight_layout()
    plt.savefig(args.output_plot_x, dpi=150)
    logger.info(f"Beam center X plot saved to {args.output_plot_x}")

    # Plot Yc_px vs Chunk_Number
    plt.figure(figsize=(10, 5))
    plt.scatter(chunk_nums, Yc_values, c='blue', marker='o', label='Yc_px', s=10)
    plt.plot(chunk_nums, Yc_values, linestyle='-', color='blue', linewidth=0.5)
    plt.axhline(median_Yc, color='green', linestyle='--', label='median Yc_px')
    plt.title('Beam Center Y Coordinate vs Chunk Number')
    plt.xlabel('Chunk Number')
    plt.ylabel('Y Coordinate (pixels)')
    plt.legend()
    plt.grid(True)
    plt.ylim(y_min_Y, y_max_Y)
    plt.tight_layout()
    plt.savefig(args.output_plot_y, dpi=150)
    logger.info(f"Beam center Y plot saved to {args.output_plot_y}")

    logger.info("Processing completed successfully.")

if __name__ == "__main__":
    main()
