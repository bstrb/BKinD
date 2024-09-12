def generate_lst_file(lst_file_name, lst_file_directory, mask_file_path, processed_h5_file_path):
    # Generate the full path for the .lst file
    lst_file_path = f"{lst_file_directory}/{lst_file_name}.lst"

    # Create the content of the .lst file
    lst_content = f"""# List of diffraction images
mask = {mask_file_path}
data = {processed_h5_file_path}
"""

    # Write the content to the .lst file
    with open(lst_file_path, 'w') as lst_file:
        lst_file.write(lst_content)

    # Output the full path of the generated .lst file
    print(f".lst file generated: {lst_file_path}")

# Example usage:
lst_file_name = "diffraction_data"
lst_file_directory = "/path/to/lst_file"
mask_file_path = "/path/to/mask_file.h5"
processed_h5_file_path = "/path/to/processed_diffraction_images.h5"

generate_lst_file(lst_file_name, lst_file_directory, mask_file_path, processed_h5_file_path)
