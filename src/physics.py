"""
physics.py
==========
Plane-Wave Expansion Method (PWEM) calculations — business/domain logic.

Preserved one-for-one from the original ``PWEM.py`` (only the background
permittivity, hardcoded to 10 in the original, became a parameter, and the
inverted material mask was corrected — see the note below). No Streamlit
imports here: everything in this module is pure numpy/scipy and fully
unit-testable.

Functions
---------
* ``build_unit_cell``    — "BUILD UNIT CELL" + "COMPUTE CONVOLUTION MATRIX"
* ``compute_band_diagram`` — "COMPUTE LIST OF BLOCH VECTORS" + "MAIN CYCLE
                             OF PROGRAM" (two generalized eigenproblems per
                             Bloch point)
"""

from __future__ import annotations

import numpy as np
from scipy import linalg

from .convmat import convmat
from .models import BandDiagram, UnitCell


def build_unit_cell(
    a: float,
    r_over_a: float,
    eps_cyl: float,
    eps_bg: float,
    nx: int,
    p_order: int,
    q_order: int,
) -> UnitCell:
    """Build the discretized unit cell and the PWEM convolution matrices.

    Mirrors the original "BUILD UNIT CELL" and "COMPUTE CONVOLUTION MATRIX"
    blocks of PWEM.py one-for-one.
    """
    r = r_over_a * a  # original: r = 0.15 * a

    # --- meshgrid on the cell (original code) -----------------------------
    Nx, Ny = nx, nx  # original: Ny = Nx
    dx = a / Nx
    dy = a / Ny
    xa = np.linspace(1, Nx, Nx) * dx
    xa = xa - np.mean(xa)
    ya = np.linspace(1, Ny, Ny) * dy
    ya = ya - np.mean(ya)
    [Y, X] = np.meshgrid(ya, xa)

    # --- build unit cell: dielectric cylinder in a background --------------
    # NOTE: the mask must be True INSIDE the cylinder (`< r*r`). The original
    # PWEM.py used `> r*r`, which filled the cylinder with eps_bg and the
    # background with eps_cyl — i.e. the two permittivities were swapped.
    UR = np.ones((Nx, Ny))
    ER = (np.square(X) + np.square(Y)) < r * r
    ER = eps_bg + (eps_cyl - eps_bg) * ER  # inside: eps_cyl, outside: eps_bg

    # --- convolution matrices (FFT based, see convmat.py) ------------------
    ERC = np.array(convmat(ER, p_order, q_order, Nx, Ny), dtype=complex)
    URC = np.array(convmat(UR, p_order, q_order, Nx, Ny), dtype=complex)

    return UnitCell(
        ER=ER,
        ERC=ERC,
        URC=URC,
        xa=xa,
        ya=ya,
        r=r,
        a=a,
        eps_cyl=eps_cyl,
        eps_bg=eps_bg,
        nx=Nx,
        p_order=p_order,
        q_order=q_order,
    )


def compute_band_diagram(
    erc: np.ndarray,
    urc: np.ndarray,
    p_order: int,
    nb: int,
    a: float,
) -> BandDiagram:
    """Sample the Bloch path and solve the E/H-mode generalized eigenproblems.

    Mirrors the original "COMPUTE LIST OF BLOCH VECTORS" and "MAIN CYCLE OF
    PROGRAM" blocks. The two matrix inverses are hoisted out of the loop —
    numerically identical to the original per-iteration ``np.linalg.inv``,
    just faster.
    """
    q_order = p_order  # original: Q = P

    # --- reciprocal lattice vectors (original code) ------------------------
    t1 = np.array([[1], [0]])
    t2 = np.array([[0], [1]])
    T1 = np.dot(t1, 2 * np.pi / a)
    T2 = np.dot(t2, 2 * np.pi / a)

    # --- main high-symmetry points of the Brillouin zone --------------------
    G = np.array([[0], [0]])
    X = np.dot(0.5, T1)
    M = np.dot(0.5, T1) + np.dot(0.5, T2)

    # --- Bloch-vector list along G->X->M->G (original code) -----------------
    GXx = np.linspace(G[0], X[0], nb)
    XMx = np.linspace(X[0], M[0], nb)
    MGx = np.linspace(M[0], G[0], int(np.floor(nb * np.sqrt(2))))

    GXy = np.linspace(G[1], X[1], nb)
    XMy = np.linspace(X[1], M[1], nb)
    MGy = np.linspace(M[1], G[1], int(np.floor(nb * np.sqrt(2))))

    # (`.ravel()` is a no-op numerically: linspace over the (2,1) high-symmetry
    # point entries returns (N,1) columns; the original script's broadcasting
    # absorbed that, but downstream pandas needs plain 1-D arrays.)
    bx = np.concatenate((GXx, XMx, MGx)).ravel()
    by = np.concatenate((GXy, XMy, MGy)).ravel()

    # --- wave vectors (original code) ---------------------------------------
    p = np.linspace(-np.floor(p_order / 2), np.floor(p_order / 2), p_order)
    q = np.linspace(-np.floor(q_order / 2), np.floor(q_order / 2), q_order)

    # Inverses needed by both eigenproblems (hoisted out of the loop)
    urc_inv = np.linalg.inv(urc)
    erc_inv = np.linalg.inv(erc)

    WnEl = []
    WnHl = []

    for ii in range(0, np.size(bx)):
        # CALCULATE WAVE VECTORS
        kx = bx[ii] - 2 * np.pi * p / a
        ky = by[ii] - 2 * np.pi * q / a

        # CREATING MESH OF WAVE VECTORS
        [ky, kx] = np.meshgrid(ky, kx)

        # CALCULATE WAVE MATRIX OF WAVE VECTORS
        KX = np.diagflat(kx)
        KY = np.diagflat(ky)

        # BUILD EIG-VALUE PROBLEM E-MODE  (A v = λ B v,  A = K·URC⁻¹·K ≈ K²)
        AE = np.dot(np.dot(KX, urc_inv), KX) + np.dot(np.dot(KY, urc_inv), KY)
        BE = erc

        # BUILD EIG-VALUE PROBLEM H-MODE  (A v = λ B v,  A = K·ERC⁻¹·K)
        AH = np.dot(np.dot(KX, erc_inv), KX) + np.dot(np.dot(KY, erc_inv), KY)
        BH = urc

        # CALCULATE EIG-VALUE
        [DEval, DEvec] = linalg.eig(AE, BE)
        [DHval, DHvec] = linalg.eig(AH, BH)

        # SORTING AND NORMALIZATION FREQUENCIES
        WnEl.append(np.sort((a * np.sqrt(DEval) / (2 * np.pi))))
        WnHl.append(np.sort((a * np.sqrt(DHval) / (2 * np.pi))))

    # CREATE AN ARRAY OF FREQUENCIES
    WnE = np.array(WnEl)
    WnH = np.array(WnHl)

    return BandDiagram(WnE=WnE, WnH=WnH, bx=bx, by=by)
