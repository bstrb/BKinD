#!/usr/bin/env python3
import sys
import math
import numpy as np

def read_xparm(path):
    """
    Minimal XPARM.XDS / GXPARM.XDS parser to get:
    - start_frame, start_angle, osc_range
    - rotation axis (lab coords)
    - wavelength, incident beam vector
    - unit cell constants
    - lab coords of unit cell a-, b-, c-axes (at spindle dial 0°)
    """
    with open(path, "r") as f:
        raw_lines = [ln.strip() for ln in f if ln.strip()]

    # XPARM.XDS files typically have a non-numeric first line ("XPARM.XDS" or similar)
    def is_float_token(tok):
        try:
            float(tok)
            return True
        except ValueError:
            return False

    # Find first line that starts with numeric tokens
    start_idx = 0
    for i, ln in enumerate(raw_lines):
        toks = ln.split()
        if toks and is_float_token(toks[0]):
            start_idx = i
            break

    lines = raw_lines[start_idx:]

    # Line 1: STARTING_FRAME, STARTING_ANGLE, OSC_RANGE, ROT_AXIS(3)
    toks1 = lines[0].split()
    if len(toks1) < 6:
        raise RuntimeError("Unexpected XPARM format in line 1")
    start_frame = int(float(toks1[0]))
    start_angle_deg = float(toks1[1])
    osc_range_deg = float(toks1[2])
    rot_axis = np.array([float(toks1[3]), float(toks1[4]), float(toks1[5])], dtype=float)

    # Line 2: WAVELENGTH, INCIDENT_BEAM_VECTOR(3)
    toks2 = lines[1].split()
    if len(toks2) < 4:
        raise RuntimeError("Unexpected XPARM format in line 2")
    wavelength = float(toks2[0])
    beam_vec = np.array([float(toks2[1]), float(toks2[2]), float(toks2[3])], dtype=float)

    # Line 3: SPACE_GROUP, a, b, c, alpha, beta, gamma
    toks3 = lines[2].split()
    if len(toks3) < 7:
        raise RuntimeError("Unexpected XPARM format in line 3")
    spacegroup = int(float(toks3[0]))
    a, b, c = float(toks3[1]), float(toks3[2]), float(toks3[3])
    alpha, beta, gamma = float(toks3[4]), float(toks3[5]), float(toks3[6])

    # Lines 4–6: UNIT_CELL_A-AXIS, B-AXIS, C-AXIS (lab coords, Å)
    a_axis = np.array([float(x) for x in lines[3].split()[:3]], dtype=float)
    b_axis = np.array([float(x) for x in lines[4].split()[:3]], dtype=float)
    c_axis = np.array([float(x) for x in lines[5].split()[:3]], dtype=float)

    return {
        "start_frame": start_frame,
        "start_angle_deg": start_angle_deg,
        "osc_range_deg": osc_range_deg,
        "rot_axis": rot_axis,
        "wavelength": wavelength,
        "beam_vec": beam_vec,
        "spacegroup": spacegroup,
        "cell": (a, b, c, alpha, beta, gamma),
        "a_axis0": a_axis,
        "b_axis0": b_axis,
        "c_axis0": c_axis,
    }

def rotation_matrix_about_axis(axis, angle_deg):
    """
    Rodrigues rotation formula.
    axis: 3-vector (will be normalized)
    angle_deg: rotation angle in degrees, right-handed.
    """
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm == 0.0:
        raise ValueError("Rotation axis has zero length")
    axis = axis / norm

    angle_rad = math.radians(angle_deg)
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    ux, uy, uz = axis

    R = np.array([
        [c + ux*ux*(1.0 - c),      ux*uy*(1.0 - c) - uz*s, ux*uz*(1.0 - c) + uy*s],
        [uy*ux*(1.0 - c) + uz*s,   c + uy*uy*(1.0 - c),    uy*uz*(1.0 - c) - ux*s],
        [uz*ux*(1.0 - c) - uy*s,   uz*uy*(1.0 - c) + ux*s, c + uz*uz*(1.0 - c)],
    ], dtype=float)
    return R

def main():
    if len(sys.argv) != 4:
        sys.stderr.write(
            "Usage: python extract_xds_orientations.py XPARM.XDS first_frame last_frame\n"
        )
        sys.exit(1)

    xparm_path = sys.argv[1]
    first_frame = int(sys.argv[2])
    last_frame = int(sys.argv[3])

    info = read_xparm(xparm_path)

    start_frame = info["start_frame"]
    start_angle_deg = info["start_angle_deg"]
    osc_range_deg = info["osc_range_deg"]
    rot_axis = info["rot_axis"]
    a0 = info["a_axis0"]
    b0 = info["b_axis0"]
    c0 = info["c_axis0"]

    # CSV header
    print("image,phi_deg,"
          "a_x,a_y,a_z,"
          "b_x,b_y,b_z,"
          "c_x,c_y,c_z")

    for img in range(first_frame, last_frame + 1):
        # XDS convention (same as your comment in the STW XDS.INP):
        # phi(i) = STARTING_ANGLE + OSCILLATION_RANGE * (i - STARTING_FRAME)
        phi_i = start_angle_deg + osc_range_deg * (img - start_frame)

        R = rotation_matrix_about_axis(rot_axis, phi_i)
        a_vec = R @ a0
        b_vec = R @ b0
        c_vec = R @ c0

        print(
            f"{img},{phi_i:.6f},"
            f"{a_vec[0]:.6f},{a_vec[1]:.6f},{a_vec[2]:.6f},"
            f"{b_vec[0]:.6f},{b_vec[1]:.6f},{b_vec[2]:.6f},"
            f"{c_vec[0]:.6f},{c_vec[1]:.6f},{c_vec[2]:.6f}"
        )

if __name__ == "__main__":
    main()
