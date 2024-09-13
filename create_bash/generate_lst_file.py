def generate_lst_file(lst_file_name, lst_file_directory, mask_file_path, processed_h5_file_paths):
    # Generate the full path for the .lst file
    lst_file_path = f"{lst_file_directory}/{lst_file_name}.lst"

    # Ensure processed_h5_file_paths is a list, even if only one file is provided
    if isinstance(processed_h5_file_paths, str):
        processed_h5_file_paths = [processed_h5_file_paths]  # Convert to a list if it's a single string

    # Create the content of the .lst file
    lst_content = f"{mask_file_path}\n"  # Add the mask file path
    lst_content += "\n".join(processed_h5_file_paths)  # Add all .h5 file paths

    # Write the content to the .lst file
    with open(lst_file_path, 'w') as lst_file:
        lst_file.write(lst_content)

    # Output the full path of the generated .lst file
    print(f".lst file generated: {lst_file_path}")
