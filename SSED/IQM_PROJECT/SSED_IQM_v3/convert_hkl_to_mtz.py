import os
import subprocess

# def convert_hkl_to_mtz(output_dir, cellfile_path):
#     """Convert the crystfel.hkl file to output.mtz using get_hkl."""
#     hkl2mtz_cmd = [
#         'get_hkl',
#         '-i', os.path.join(output_dir, "crystfel.hkl"),
#         '-o', os.path.join(output_dir, "output.mtz"),
#         '-p', f'{cellfile_path}',
#         '--output-format=mtz'
#     ]

#     try:
#         with open(os.path.join(output_dir, "stdout.log"), "a") as stdout, open(os.path.join(output_dir, "stderr.log"), "a") as stderr:
#             print(f"Converting crystfel.hkl to output.mtz in directory: {output_dir}")
#             subprocess.run(hkl2mtz_cmd, stdout=stdout, stderr=stderr, check=True)
#             print(f"Conversion to output.mtz completed for directory: {output_dir}")
#     except subprocess.CalledProcessError as e:
#         print(f"Error during conversion to MTZ in {output_dir}: {e}")
#         raise

def convert_hkl_to_mtz(output_dir, cellfile_path):
    """Convert the crystfel.hkl file to output.mtz using get_hkl."""
    input_hkl = os.path.join(output_dir, "crystfel.hkl")
    output_mtz = os.path.join(output_dir, "output.mtz")
    hkl2mtz_cmd = (
        "get_hkl "
        f"-i {input_hkl} "
        f"-o {output_mtz} "
        f"-p {cellfile_path} "
        "--output-format=mtz"
    )

    try:
        with open(os.path.join(output_dir, "stdout.log"), "a") as stdout, open(os.path.join(output_dir, "stderr.log"), "a") as stderr:
            print(f"Converting crystfel.hkl to output.mtz in directory: {output_dir}")
            subprocess.run(hkl2mtz_cmd, stdout=stdout, stderr=stderr, check=True)
            print(f"Conversion to output.mtz completed for directory: {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion to MTZ in {output_dir}: {e}")
        raise
