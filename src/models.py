"""
models.py
=========
Data structures shared between the physics engine (``src.physics``) and the
Streamlit UI (``app.py``).

Plain ``dataclasses`` (no external dependencies):

* ``UnitCell``    — discretized unit cell + FFT convolution matrices
* ``BandDiagram`` — eigenfrequencies sampled along the Bloch path Γ→X→M→Γ
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class UnitCell:
    """A discretized square-lattice unit cell and its PWEM matrices.

    Attributes
    ----------
    ER      : (Nx, Ny) real array — permittivity grid (eps_cyl inside the
              cylinder, eps_bg outside)
    ERC     : (P·Q, P·Q) complex array — FFT convolution matrix of ε
    URC     : (P·Q, P·Q) complex array — FFT convolution matrix of 1 (≈ I)
    xa, ya  : (Nx,)/ (Ny,) coordinate vectors, centered at the cell origin
    r       : cylinder radius (absolute units)
    a       : lattice constant
    eps_cyl : cylinder permittivity
    eps_bg  : background permittivity
    nx      : grid size (Nx = Ny)
    p_order : plane-wave order along x (P)
    q_order : plane-wave order along y (Q)
    """

    ER: np.ndarray
    ERC: np.ndarray
    URC: np.ndarray
    xa: np.ndarray
    ya: np.ndarray
    r: float
    a: float
    eps_cyl: float
    eps_bg: float
    nx: int
    p_order: int
    q_order: int

    @property
    def n_plane_waves(self) -> int:
        """Total number of plane-wave basis vectors (P × Q)."""
        return self.p_order * self.q_order


@dataclass
class BandDiagram:
    """Eigenfrequencies (ωa/2πc) along the high-symmetry path Γ→X→M→Γ.

    Attributes
    ----------
    WnE, WnH : (n_points, P·Q) arrays of normalized frequencies for the
               E-mode (TE) and H-mode (TM) polarization, sorted ascending
               along the last axis
    bx, by   : (n_points,) Bloch-wave-vector components of the path samples
    """

    WnE: np.ndarray
    WnH: np.ndarray
    bx: np.ndarray
    by: np.ndarray

    @property
    def n_points(self) -> int:
        """Number of sampled Bloch points on the path."""
        return int(np.size(self.bx))
