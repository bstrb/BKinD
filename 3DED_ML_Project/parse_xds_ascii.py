# parse_xds_ascii.py

import numpy as np

def parse_xds_ascii(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('!END_OF_HEADER'):
                break
        for line in f:
            if line.startswith('!END_OF_DATA'):
                break
            fields = line.split()
            hkl = tuple(map(int, fields[:3]))
            iobs = float(fields[3])
            sigma_iobs = float(fields[4])
            xd, yd, zd = map(float, fields[5:8])
            rlp = float(fields[8])
            peak = int(fields[9])
            corr = int(fields[10])
            psi = float(fields[11])
            data.append([hkl, iobs, sigma_iobs, xd, yd, zd, rlp, peak, corr, psi])
    return np.array(data, dtype=object)

if __name__ == "__main__":
    data = parse_xds_ascii('XDS_ASCII.HKL')
    print(data[:5])
