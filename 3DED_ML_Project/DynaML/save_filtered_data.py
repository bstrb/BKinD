# save_filtered_data.py
def save_filtered_data(header_lines, df, output_file_path):
    with open(output_file_path, 'w') as file:
        # Write the header lines
        file.write(''.join(header_lines))

        # Write filtered reflections
        for _, row in df.iterrows():
            # Ensure that h, k, l are integers
            h = int(row['h'])
            k = int(row['k'])
            l = int(row['l'])
            file.write(f"{h:4d} {k:4d} {l:4d} {row['iobs']:12.4e} "
                       f"{row['sigma_iobs']:12.4e} {row['xd']:8.1f} {row['yd']:8.1f} {row['zd']:8.1f} "
                       f"{row['rlp']:8.4f} {int(row['peak']):4d} {int(row['corr']):4d} {row['psi']:8.2f} "
                       f"{row['cbi']:12.6f}\n")
        
        # Add the end of data marker
        file.write("!END_OF_DATA\n")
