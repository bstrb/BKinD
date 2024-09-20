def generate_lst_file(lst_file_name, lst_file_directory, source_h5_file_paths):
    # Generate the full path for the .lst file
    lst_file_path = f"{lst_file_directory}/{lst_file_name}.lst"

    # Ensure source_h5_file_paths is a list
    if isinstance(source_h5_file_paths, str):
        source_h5_file_paths = [source_h5_file_paths]

    # Create the content of the .lst file
    lst_content = "\n".join(source_h5_file_paths)  # Add all .h5 file paths

    # Write the content to the .lst file
    with open(lst_file_path, 'w') as lst_file:
        lst_file.write(lst_content)

    # Output the full path of the generated .lst file
    print(f".lst file generated: {lst_file_path}")
