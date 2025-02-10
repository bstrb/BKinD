# run_indexamajig.py

import os
import subprocess


def run_indexamajig(x, y, geomfile_path, cellfile_path, input_path, output_file_base, output_dir, num_threads, indexing_method, resolution_push, integration_method, int_radius, min_peaks, xgt, xgsp, xgi, tolerance):

    output_file = f"{output_file_base}_{x}_{y}.stream"
    output_path = os.path.join(output_dir, output_file)
    listfile_path = os.path.join(input_path, 'list.lst')

    base_command = (
        f"indexamajig -g {geomfile_path} -i {listfile_path} -o {output_path} -p {cellfile_path} "
        f"-j {num_threads} --indexing={indexing_method} --integration={integration_method}  --int-radius={int_radius} " # --no-check-cell-nocell-nolatt
        f"--tolerance={tolerance} --min-peaks={min_peaks} " #--xgandalf-max-peaks=150 --push-res={resolution_push}
        f"--peaks=peakfinder9 --peak-radius=4.0,5.0,7.0 --min-sig=25 --min-snr-biggest-pix=1 --min-snr=1 --local-bg-radius=1 " # --peaks=cxi
        f"--xgandalf-sampling-pitch={xgsp} --xgandalf-grad-desc-iterations={xgi} --xgandalf-tolerance={xgt} " 
        f"--no-half-pixel-shift --no-non-hits-in-stream " # --no-image-data  
        f"--no-refine --no-revalidate " # --no-check-cell --no-check-peaks --no-retry
    )

    subprocess.run(base_command, shell=True, check=True)

