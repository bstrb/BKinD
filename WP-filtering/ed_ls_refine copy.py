#!/usr/bin/env python3
"""
ed_ls_refine.py  (FIXED SIGN)

Minimal SHELXL-like least-squares refiner for electron diffraction against Fo^2 (HKLF 4).

Fix vs previous version:
- LM/Gauss-Newton step now solves (J^T J + lam*D) dp = -J^T r  (minus sign!)
  Previously it solved +J^T r, which makes every step go uphill -> always rejected.

Other behavior unchanged.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


_TWO_PI = 2.0 * math.pi


def is_close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


ATOMIC_Z: Dict[str, int] = {
    "H": 1, "C": 6, "N": 7, "O": 8, "F": 9,
    "SI": 14, "P": 15, "S": 16, "CL": 17,
    "NA": 11, "MG": 12, "AL": 13, "K": 19, "CA": 20,
    "FE": 26, "CU": 29, "ZN": 30, "BR": 35, "I": 53,
}


@dataclass
class Cell:
    wavelength: float
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float

    def is_orthogonal(self, tol_deg: float = 1e-4) -> bool:
        return (abs(self.alpha - 90.0) < tol_deg and
                abs(self.beta - 90.0) < tol_deg and
                abs(self.gamma - 90.0) < tol_deg)

    def reciprocal_lengths(self) -> Tuple[float, float, float]:
        if not self.is_orthogonal():
            raise ValueError("reciprocal_lengths() supports only orthogonal cells (90/90/90).")
        return 1.0 / self.a, 1.0 / self.b, 1.0 / self.c

    def q_vector_cart(self, h: int, k: int, l: int) -> np.ndarray:
        ast, bst, cst = self.reciprocal_lengths()
        return np.array([h * ast, k * bst, l * cst], dtype=float)

    def s_sin_theta_over_lambda(self, h: int, k: int, l: int) -> float:
        q = self.q_vector_cart(h, k, l)
        qmag = float(np.linalg.norm(q))
        return qmag / (4.0 * math.pi)


@dataclass
class Scatterer:
    label: str
    a: np.ndarray  # (4,)
    b: np.ndarray  # (4,)
    c: float

    def f0(self, s: float) -> float:
        s2 = s * s
        return float(np.sum(self.a * np.exp(-self.b * s2)) + self.c)


@dataclass
class Atom:
    name: str
    sfac_index: int
    x: float
    y: float
    z: float
    sof_raw: float
    u_raw: List[float]
    occ_kind: str
    occ_multiplier: float
    occ_fvar_index: Optional[int]

    def frac_pos(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)


@dataclass
class Reflection:
    h: int
    k: int
    l: int
    fo2: float
    sig: float


@dataclass
class Model:
    cell: Cell
    scatterers: List[Scatterer]
    atoms: List[Atom]
    fvars: List[float]   # fvars[0] is scale k
    wght_a: float
    wght_b: float
    anis: bool


@dataclass
class ParamMap:
    idx_k: int
    idx_fvar: Dict[int, int]
    idx_atom_xyz: Dict[int, Tuple[int, int, int]]
    idx_atom_u: Dict[int, List[int]]
    idx_atom_occ_lit: Dict[int, int]


_INSTR_RE = re.compile(r"^\s*([A-Za-z]{4})\b")


def _strip_comment(line: str) -> str:
    if "!" in line:
        line = line.split("!", 1)[0]
    return line.strip()


def _looks_numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def decode_sof(sof: float) -> Tuple[str, float, Optional[int]]:
    m = int(sof / 10.0)
    p = sof - 10.0 * m
    if m != 0 and abs(p) < 5.0:
        if m == 1:
            return "fixed", float(p), None
        if m > 1:
            return "fvar", float(p), m
        if m < -1:
            return "fvar_minus1", float(p), -m
    return "literal", float(sof), None


def parse_ins(path: str) -> Model:
    cell: Optional[Cell] = None
    sfac_tokens: List[str] = []
    scatterers: List[Scatterer] = []
    atoms: List[Atom] = []
    fvars: Optional[List[float]] = None
    wght_a = 0.0
    wght_b = 0.0
    anis = False

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    for raw in lines:
        line = _strip_comment(raw)
        if not line:
            continue
        if line.upper().startswith("REM"):
            continue

        m = _INSTR_RE.match(line)
        key = m.group(1).upper() if m else ""
        parts = line.split()

        if key == "CELL":
            if len(parts) < 8:
                raise ValueError(f"Bad CELL line: {line}")
            lam = float(parts[1])
            a, b, c = float(parts[2]), float(parts[3]), float(parts[4])
            al, be, ga = float(parts[5]), float(parts[6]), float(parts[7])
            cell = Cell(lam, a, b, c, al, be, ga)

        elif key == "SFAC":
            if len(parts) < 2:
                raise ValueError(f"Bad SFAC line: {line}")
            if len(parts) == 2 or (len(parts) > 2 and not _looks_numeric(parts[2])):
                for tok in parts[1:]:
                    sfac_tokens.append(tok.strip())
            else:
                label = parts[1].strip()
                nums = [float(x) for x in parts[2:]]
                if len(nums) < 9:
                    raise ValueError(f"SFAC coeff form requires >=9 numbers after label: {line}")
                aco = np.array([nums[0], nums[2], nums[4], nums[6]], dtype=float)
                bco = np.array([nums[1], nums[3], nums[5], nums[7]], dtype=float)
                c0 = float(nums[8])
                scatterers.append(Scatterer(label=label.upper(), a=aco, b=bco, c=c0))

        elif key == "FVAR":
            if len(parts) < 2:
                raise ValueError(f"Bad FVAR line: {line}")
            fvars = [float(x) for x in parts[1:]]

        elif key == "WGHT":
            if len(parts) >= 3:
                wght_a = float(parts[1])
                wght_b = float(parts[2])

        elif key == "ANIS":
            anis = True

        elif key == "HKLF":
            break

        elif key in {"TITL", "UNIT", "LATT", "SYMM", "ZERR", "L.S.", "LIST", "MERG", "SHEL", "MORE", "FMAP", "PLAN", "CONF", "END"}:
            continue

        else:
            if len(parts) < 7:
                continue
            name = parts[0]
            try:
                sfac_idx = int(parts[1])
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                sof = float(parts[5])
                rest = [float(x) for x in parts[6:]]
            except ValueError:
                continue

            if anis:
                if len(rest) < 6:
                    uiso = float(rest[0])
                    u_raw = [uiso, uiso, uiso, 0.0, 0.0, 0.0]
                else:
                    u_raw = rest[:6]
            else:
                u_raw = [float(rest[0])]

            occ_kind, occ_mul, occ_fv = decode_sof(sof)
            atoms.append(Atom(
                name=name,
                sfac_index=sfac_idx,
                x=x, y=y, z=z,
                sof_raw=sof,
                u_raw=u_raw,
                occ_kind=occ_kind,
                occ_multiplier=occ_mul,
                occ_fvar_index=occ_fv,
            ))

    if cell is None:
        raise ValueError("Missing CELL in .ins")
    if fvars is None:
        fvars = [1.0]

    if sfac_tokens:
        label_to_sc = {sc.label.upper(): sc for sc in scatterers}
        scatterers_final: List[Scatterer] = []
        for tok in sfac_tokens:
            lab = tok.strip().upper()
            if lab in label_to_sc:
                scatterers_final.append(label_to_sc[lab])
            else:
                z = ATOMIC_Z.get(lab, None)
                if z is None:
                    raise ValueError(f"SFAC symbol '{lab}' not in fallback Z table; provide coeff-form SFAC.")
                aco = np.array([float(z), 0.0, 0.0, 0.0], dtype=float)
                bco = np.array([0.0, 1.0, 1.0, 1.0], dtype=float)
                scatterers_final.append(Scatterer(label=lab, a=aco, b=bco, c=0.0))
        scatterers = scatterers_final
    else:
        if not scatterers:
            raise ValueError("No SFAC provided.")

    return Model(
        cell=cell,
        scatterers=scatterers,
        atoms=atoms,
        fvars=fvars,
        wght_a=wght_a,
        wght_b=wght_b,
        anis=anis,
    )


def read_hkl_hklf4(path: str) -> List[Reflection]:
    refs: List[Reflection] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                h, k, l = int(parts[0]), int(parts[1]), int(parts[2])
                fo2 = float(parts[3])
                sig = float(parts[4])
            except ValueError:
                continue
            if h == 0 and k == 0 and l == 0 and fo2 == 0 and sig == 0:
                break
            if sig <= 0:
                sig = 1e-6
            refs.append(Reflection(h=h, k=k, l=l, fo2=fo2, sig=sig))
    if not refs:
        raise ValueError("No reflections read from HKLF4 file.")
    return refs


def build_param_vector_and_map(model: Model) -> Tuple[np.ndarray, ParamMap]:
    p: List[float] = []
    idx_k = 0
    p.append(model.fvars[0])

    used_fv: List[int] = []
    for a in model.atoms:
        if a.occ_kind in {"fvar", "fvar_minus1"} and a.occ_fvar_index is not None:
            used_fv.append(a.occ_fvar_index)
    used_fv = sorted(set([i for i in used_fv if i >= 2]))

    idx_fvar: Dict[int, int] = {}
    for fv_i in used_fv:
        while fv_i - 1 >= len(model.fvars):
            model.fvars.append(1.0)
        idx_fvar[fv_i] = len(p)
        p.append(model.fvars[fv_i - 1])

    idx_atom_xyz: Dict[int, Tuple[int, int, int]] = {}
    idx_atom_u: Dict[int, List[int]] = {}
    idx_atom_occ_lit: Dict[int, int] = {}

    for i, a in enumerate(model.atoms):
        ix = len(p); p.append(a.x)
        iy = len(p); p.append(a.y)
        iz = len(p); p.append(a.z)
        idx_atom_xyz[i] = (ix, iy, iz)

        if model.anis:
            iu = []
            uvals = a.u_raw if len(a.u_raw) == 6 else [a.u_raw[0], a.u_raw[0], a.u_raw[0], 0.0, 0.0, 0.0]
            for uv in uvals:
                iu.append(len(p)); p.append(float(uv))
            idx_atom_u[i] = iu
        else:
            iu0 = len(p); p.append(float(a.u_raw[0]))
            idx_atom_u[i] = [iu0]

        if a.occ_kind == "literal":
            io = len(p); p.append(float(a.occ_multiplier))
            idx_atom_occ_lit[i] = io

    return np.array(p, dtype=float), ParamMap(
        idx_k=idx_k,
        idx_fvar=idx_fvar,
        idx_atom_xyz=idx_atom_xyz,
        idx_atom_u=idx_atom_u,
        idx_atom_occ_lit=idx_atom_occ_lit,
    )


def unpack_params_into_model(model: Model, p: np.ndarray, pm: ParamMap) -> None:
    model.fvars[0] = float(p[pm.idx_k])
    for fv_i, pi in pm.idx_fvar.items():
        model.fvars[fv_i - 1] = float(p[pi])

    for i, a in enumerate(model.atoms):
        ix, iy, iz = pm.idx_atom_xyz[i]
        a.x = float(p[ix]); a.y = float(p[iy]); a.z = float(p[iz])

        uidx = pm.idx_atom_u[i]
        if model.anis:
            a.u_raw = [float(p[j]) for j in uidx]
        else:
            a.u_raw = [float(p[uidx[0]])]

        if i in pm.idx_atom_occ_lit:
            a.occ_multiplier = float(p[pm.idx_atom_occ_lit[i]])


def atom_occupancy(model: Model, atom: Atom) -> float:
    if atom.occ_kind in {"fixed", "literal"}:
        return float(atom.occ_multiplier)
    if atom.occ_kind == "fvar":
        fv = model.fvars[atom.occ_fvar_index - 1]  # type: ignore
        return float(atom.occ_multiplier * fv)
    if atom.occ_kind == "fvar_minus1":
        fv = model.fvars[atom.occ_fvar_index - 1]  # type: ignore
        return float(atom.occ_multiplier * (fv - 1.0))
    raise ValueError(f"Unknown occ_kind: {atom.occ_kind}")


def scattering_factor(model: Model, atom: Atom, s: float) -> float:
    sc = model.scatterers[atom.sfac_index - 1]
    return sc.f0(s)


def dw_factor_iso(U: float, s: float) -> float:
    return math.exp(-8.0 * (math.pi ** 2) * U * (s * s))


def dw_factor_anis_orthogonal(cell: Cell, h: int, k: int, l: int, Uij: List[float]) -> float:
    ast, bst, cst = cell.reciprocal_lengths()
    U11, U22, U33, U23, U13, U12 = Uij
    Q = (h*h*(ast*ast)*U11 +
         k*k*(bst*bst)*U22 +
         l*l*(cst*cst)*U33 +
         2.0*h*k*ast*bst*U12 +
         2.0*h*l*ast*cst*U13 +
         2.0*k*l*bst*cst*U23)
    return math.exp(-2.0 * (math.pi ** 2) * Q)


def compute_Fc_and_partials(model: Model, refl: Reflection) -> Tuple[complex, List[complex], float]:
    h, k, l = refl.h, refl.k, refl.l
    s = model.cell.s_sin_theta_over_lambda(h, k, l)

    Bj: List[complex] = []
    Fc: complex = 0.0 + 0.0j
    for a in model.atoms:
        occ = clamp(atom_occupancy(model, a), -2.0, 2.0)
        f0 = scattering_factor(model, a, s)

        phi = _TWO_PI * (h * a.x + k * a.y + l * a.z)
        phase = complex(math.cos(phi), math.sin(phi))

        if model.anis:
            if not model.cell.is_orthogonal():
                raise ValueError("ANIS requires orthogonal cell (90/90/90) in this MVP.")
            Uij = a.u_raw if len(a.u_raw) == 6 else [a.u_raw[0], a.u_raw[0], a.u_raw[0], 0.0, 0.0, 0.0]
            T = dw_factor_anis_orthogonal(model.cell, h, k, l, Uij)
        else:
            T = dw_factor_iso(a.u_raw[0], s)

        bj = (f0 * T) * phase
        Bj.append(bj)
        Fc += occ * bj

    return Fc, Bj, s


def weight_wght(model: Model, fo2: float, sig: float, fc2: float) -> float:
    a = model.wght_a
    b = model.wght_b
    if is_close(a, 0.0) and is_close(b, 0.0):
        return 1.0 / (sig * sig)
    P = (max(fo2, 0.0) + 2.0 * fc2) / 3.0
    denom = (sig * sig) + (a * P) ** 2 + (b * P)
    if denom <= 0:
        denom = 1e-12
    return 1.0 / denom


def refine(model: Model, refls: List[Reflection], cycles: int, lm_lambda: float, verbose: bool = True) -> np.ndarray:
    if model.anis and not model.cell.is_orthogonal():
        raise ValueError("This MVP supports ANIS only for orthogonal cells (90/90/90).")

    p, pm = build_param_vector_and_map(model)

    lam = lm_lambda
    for cyc in range(1, cycles + 1):
        unpack_params_into_model(model, p, pm)

        model.fvars[0] = max(model.fvars[0], 1e-12)
        for a in model.atoms:
            if model.anis:
                a.u_raw[0] = max(a.u_raw[0], 1e-8)
                a.u_raw[1] = max(a.u_raw[1], 1e-8)
                a.u_raw[2] = max(a.u_raw[2], 1e-8)
                a.u_raw[3] = clamp(a.u_raw[3], -0.1, 0.1)
                a.u_raw[4] = clamp(a.u_raw[4], -0.1, 0.1)
                a.u_raw[5] = clamp(a.u_raw[5], -0.1, 0.1)
            else:
                a.u_raw[0] = max(a.u_raw[0], 1e-8)

        m = len(refls)
        n = len(p)
        r = np.zeros(m, dtype=float)
        J = np.zeros((m, n), dtype=float)

        k_scale = model.fvars[0]
        wssq = 0.0

        for iref, ref in enumerate(refls):
            Fc, Bj, s = compute_Fc_and_partials(model, ref)
            fc2_no_scale = (Fc.real * Fc.real + Fc.imag * Fc.imag)
            Ic = k_scale * fc2_no_scale

            w = weight_wght(model, ref.fo2, ref.sig, Ic)
            sw = math.sqrt(w)

            ri = sw * (ref.fo2 - Ic)
            r[iref] = ri
            wssq += w * (ref.fo2 - Ic) ** 2

            dI_dk = fc2_no_scale
            J[iref, pm.idx_k] = -sw * dI_dk

            F_conj = complex(Fc.real, -Fc.imag)

            for fv_i, pidx in pm.idx_fvar.items():
                dF = 0.0 + 0.0j
                for ia, a in enumerate(model.atoms):
                    if a.occ_kind in {"fvar", "fvar_minus1"} and a.occ_fvar_index == fv_i:
                        d_occ = a.occ_multiplier
                        dF += d_occ * Bj[ia]
                if dF != 0.0j:
                    dI = k_scale * 2.0 * (F_conj.real * dF.real - F_conj.imag * dF.imag)
                    J[iref, pidx] = -sw * dI

            for ia, a in enumerate(model.atoms):
                ix, iy, iz = pm.idx_atom_xyz[ia]
                occ = clamp(atom_occupancy(model, a), -2.0, 2.0)
                bj = Bj[ia]

                dF_dx = occ * bj * (complex(0.0, _TWO_PI * ref.h))
                dF_dy = occ * bj * (complex(0.0, _TWO_PI * ref.k))
                dF_dz = occ * bj * (complex(0.0, _TWO_PI * ref.l))

                dI_dx = k_scale * 2.0 * (F_conj.real * dF_dx.real - F_conj.imag * dF_dx.imag)
                dI_dy = k_scale * 2.0 * (F_conj.real * dF_dy.real - F_conj.imag * dF_dy.imag)
                dI_dz = k_scale * 2.0 * (F_conj.real * dF_dz.real - F_conj.imag * dF_dz.imag)

                J[iref, ix] = -sw * dI_dx
                J[iref, iy] = -sw * dI_dy
                J[iref, iz] = -sw * dI_dz

                if ia in pm.idx_atom_occ_lit:
                    io = pm.idx_atom_occ_lit[ia]
                    dF_doc = bj
                    dI_doc = k_scale * 2.0 * (F_conj.real * dF_doc.real - F_conj.imag * dF_doc.imag)
                    J[iref, io] = -sw * dI_doc

                uidx = pm.idx_atom_u[ia]
                if model.anis:
                    ast, bst, cst = model.cell.reciprocal_lengths()
                    h, k, l = ref.h, ref.k, ref.l
                    dE = [
                        -2.0*(math.pi**2) * (h*h*(ast*ast)),
                        -2.0*(math.pi**2) * (k*k*(bst*bst)),
                        -2.0*(math.pi**2) * (l*l*(cst*cst)),
                        -2.0*(math.pi**2) * (2.0*k*l*bst*cst),
                        -2.0*(math.pi**2) * (2.0*h*l*ast*cst),
                        -2.0*(math.pi**2) * (2.0*h*k*ast*bst),
                    ]
                    for j, pj in enumerate(uidx):
                        dF = occ * bj * dE[j]
                        dI = k_scale * 2.0 * (F_conj.real * dF.real - F_conj.imag * dF.imag)
                        J[iref, pj] = -sw * dI
                else:
                    pj = uidx[0]
                    dE = -8.0 * (math.pi ** 2) * (s * s)
                    dF = occ * bj * dE
                    dI = k_scale * 2.0 * (F_conj.real * dF.real - F_conj.imag * dF.imag)
                    J[iref, pj] = -sw * dI

        JTJ = J.T @ J
        JTr = J.T @ r

        D = np.diag(np.diag(JTJ))
        A = JTJ + lam * (D + 1e-12 * np.eye(n))

        # *** FIXED SIGN HERE ***
        try:
            dp = -np.linalg.solve(A, JTr)
        except np.linalg.LinAlgError:
            dp = -np.linalg.lstsq(A, JTr, rcond=None)[0]

        p_new = p + dp

        # Evaluate trial objective
        unpack_params_into_model(model, p_new, pm)
        model.fvars[0] = max(model.fvars[0], 1e-12)
        for a in model.atoms:
            if model.anis:
                a.u_raw[0] = max(a.u_raw[0], 1e-8)
                a.u_raw[1] = max(a.u_raw[1], 1e-8)
                a.u_raw[2] = max(a.u_raw[2], 1e-8)
                a.u_raw[3] = clamp(a.u_raw[3], -0.1, 0.1)
                a.u_raw[4] = clamp(a.u_raw[4], -0.1, 0.1)
                a.u_raw[5] = clamp(a.u_raw[5], -0.1, 0.1)
            else:
                a.u_raw[0] = max(a.u_raw[0], 1e-8)

        wssq_new = 0.0
        for ref in refls:
            Fc, _, _ = compute_Fc_and_partials(model, ref)
            fc2 = (Fc.real * Fc.real + Fc.imag * Fc.imag)
            Ic = model.fvars[0] * fc2
            w = weight_wght(model, ref.fo2, ref.sig, Ic)
            wssq_new += w * (ref.fo2 - Ic) ** 2

        accept = (wssq_new < wssq)

        if accept:
            p = p_new
            lam = max(lam * 0.5, 1e-12)
        else:
            unpack_params_into_model(model, p, pm)
            lam = min(lam * 5.0, 1e12)

        if verbose:
            print(f"Cycle {cyc:3d}  wSSQ={wssq: .6e}  trial={wssq_new: .6e}  accept={accept}  lm={lam: .3e}")
            if np.linalg.norm(dp) < 1e-10:
                print("Converged (step norm small).")
                break

    unpack_params_into_model(model, p, pm)
    return p


def write_ins_like(model: Model, out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        c = model.cell
        f.write("TITL refined_by_ed_ls_refine\n")
        f.write(f"CELL {c.wavelength:.8f} {c.a:.6f} {c.b:.6f} {c.c:.6f} {c.alpha:.4f} {c.beta:.4f} {c.gamma:.4f}\n")
        f.write("SFAC " + " ".join([sc.label for sc in model.scatterers]) + "\n")
        for sc in model.scatterers:
            nums = [sc.a[0], sc.b[0], sc.a[1], sc.b[1], sc.a[2], sc.b[2], sc.a[3], sc.b[3], sc.c]
            f.write("SFAC " + sc.label + " " + " ".join(f"{x:.6g}" for x in nums) + "\n")
        f.write("FVAR " + " ".join(f"{x:.10g}" for x in model.fvars) + "\n")
        f.write(f"WGHT {model.wght_a:.6g} {model.wght_b:.6g}\n")
        if model.anis:
            f.write("ANIS\n")
            f.write("! name sfac x y z sof U11 U22 U33 U23 U13 U12\n")
        else:
            f.write("! name sfac x y z sof Uiso\n")

        for a in model.atoms:
            if model.anis:
                u = a.u_raw if len(a.u_raw) == 6 else [a.u_raw[0]]*3 + [0.0, 0.0, 0.0]
                f.write(f"{a.name:<4s} {a.sfac_index:2d} {a.x: .6f} {a.y: .6f} {a.z: .6f} {a.sof_raw: .6f} " +
                        " ".join(f"{x: .6g}" for x in u) + "\n")
            else:
                f.write(f"{a.name:<4s} {a.sfac_index:2d} {a.x: .6f} {a.y: .6f} {a.z: .6f} {a.sof_raw: .6f} {a.u_raw[0]: .6g}\n")

        f.write("HKLF 4\nEND\n")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ins", help="Input .ins (minimal subset)")
    ap.add_argument("hkl", help="HKLF4 reflection file (h k l Fo2 sig)")
    ap.add_argument("--cycles", type=int, default=30, help="Max refinement cycles")
    ap.add_argument("--lm", type=float, default=1e-2, help="Initial Levenberg-Marquardt damping")
    ap.add_argument("--out", type=str, default="", help="Write refined .ins-like file")
    ap.add_argument("--quiet", action="store_true", help="Less printing")
    args = ap.parse_args()

    model = parse_ins(args.ins)
    refls = read_hkl_hklf4(args.hkl)

    if model.anis and not model.cell.is_orthogonal():
        raise SystemExit("ERROR: ANIS is enabled but cell is not orthogonal (alpha/beta/gamma != 90).")

    print(f"Read {len(model.atoms)} atoms, {len(model.scatterers)} SFAC scatterers, {len(refls)} reflections.")
    print(f"Refining with WGHT a={model.wght_a} b={model.wght_b} ; ANIS={model.anis}")
    print(f"Initial scale k = {model.fvars[0]}")

    refine(model, refls, cycles=args.cycles, lm_lambda=args.lm, verbose=(not args.quiet))

    print("\nFinal parameters:")
    print(f"  scale k = {model.fvars[0]:.10g}")
    print(f"  sqrt scale k = {model.fvars[0]**0.5:.10g}")
    if len(model.fvars) > 1:
        for i in range(2, len(model.fvars) + 1):
            print(f"  FVAR[{i}] = {model.fvars[i-1]:.10g}")

    for a in model.atoms:
        occ = atom_occupancy(model, a)
        if model.anis:
            u = a.u_raw if len(a.u_raw) == 6 else [a.u_raw[0]]*3 + [0.0, 0.0, 0.0]
            print(f"  {a.name:>4s} occ={occ: .6f} xyz=({a.x: .6f},{a.y: .6f},{a.z: .6f}) "
                  f"U11..U12=({u[0]:.4g},{u[1]:.4g},{u[2]:.4g},{u[3]:.4g},{u[4]:.4g},{u[5]:.4g})")
        else:
            print(f"  {a.name:>4s} occ={occ: .6f} xyz=({a.x: .6f},{a.y: .6f},{a.z: .6f}) Uiso={a.u_raw[0]:.6g}")

    if args.out:
        write_ins_like(model, args.out)
        print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()

#   python ed_ls_refine.py /Users/xiaodong/Desktop/3DED-DATA/LTA/LTA4/wght_study/wght_study.ins /Users/xiaodong/Desktop/3DED-DATA/LTA/LTA4/wght_study/wght_study.hkl --cycles 30 --lm 1e-2