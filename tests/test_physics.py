"""
test_physics.py — unit tests for the PWEM engine (src/physics).

Small grids / low plane-wave orders keep the suite fast (< a few seconds).
The high-contrast test doubles as a regression guard for the material-mask
fix: the band gap must appear in the E-mode (TE) of high-ε rods in air.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models import BandDiagram, UnitCell
from src.physics import build_unit_cell, compute_band_diagram


def _small_cell() -> UnitCell:
    return build_unit_cell(
        a=1.0, r_over_a=0.15, eps_cyl=2.0, eps_bg=10.0,
        nx=33, p_order=5, q_order=5,
    )


# --- unit cell -----------------------------------------------------------------


def test_unit_cell_mask_assigns_permittivities_correctly():
    """Inside the cylinder: eps_cyl; outside: eps_bg (the swapped-mask bug)."""
    cell = _small_cell()
    c = cell.nx // 2
    assert cell.ER[c, c] == pytest.approx(cell.eps_cyl)
    assert cell.ER[0, 0] == pytest.approx(cell.eps_bg)


def test_unit_cell_shapes_and_radius():
    cell = _small_cell()
    assert cell.ER.shape == (33, 33)
    assert cell.ERC.shape == (25, 25)   # P*Q x P*Q
    assert cell.URC.shape == (25, 25)
    assert cell.xa.size == 33 and cell.ya.size == 33
    assert cell.r == pytest.approx(0.15)
    assert cell.n_plane_waves == 25


def test_urc_of_ones_is_identity():
    """The convolution matrix of the constant-1 function is (≈) the identity."""
    cell = _small_cell()
    assert np.allclose(cell.URC, np.eye(25), atol=1e-12)


def test_erc_dc_block_is_permittivity_mean():
    """The (0,0) entry of the FFT-convolution matrix is the mean value."""
    cell = _small_cell()
    # A[0,0] of the FFT-convolution matrix is the zero-frequency (mean) component
    assert cell.ERC[0, 0].real == pytest.approx(np.mean(cell.ER), rel=1e-12)


# --- band diagram ----------------------------------------------------------------


def test_band_diagram_shapes_and_ordering():
    cell = _small_cell()
    bands = compute_band_diagram(cell.ERC, cell.URC, p_order=5, nb=5, a=1.0)
    assert isinstance(bands, BandDiagram)
    n_points = 5 + 5 + int(np.floor(5 * np.sqrt(2)))
    assert bands.n_points == n_points
    assert bands.WnE.shape == (n_points, 25)
    assert bands.WnH.shape == (n_points, 25)
    assert bands.bx.ndim == 1 and bands.by.ndim == 1
    assert bands.bx.size == n_points
    # eigenfrequencies sorted ascending and non-negative
    assert np.all(np.diff(np.real(bands.WnE), axis=1) >= -1e-9)
    assert np.all(np.diff(np.real(bands.WnH), axis=1) >= -1e-9)
    assert np.real(bands.WnE).min() >= -1e-9
    assert np.real(bands.WnH).min() >= -1e-9


def test_high_contrast_rods_open_the_te_gap():
    """Regression test for the material-mask fix.

    Silicon rods (eps=11.7) in air (eps=1.0) must open a band gap between the
    1st and 2nd E-mode (TE) bands; the dual H-mode (TM) problem has no such
    gap — the exact configuration the original (inverted-mask) code computed
    backwards.
    """
    cell = build_unit_cell(
        a=1.55, r_over_a=0.148, eps_cyl=11.7, eps_bg=1.0,
        nx=129, p_order=9, q_order=9,
    )
    bands = compute_band_diagram(cell.ERC, cell.URC, p_order=9, nb=15, a=1.55)

    e1, e2 = np.real(bands.WnE)[:, 0], np.real(bands.WnE)[:, 1]
    h1, h2 = np.real(bands.WnH)[:, 0], np.real(bands.WnH)[:, 1]

    # E-mode (TE): a genuine gap — 1st band max < 2nd band min
    assert e1.max() < e2.min()
    # H-mode (TM): the bands cross — no gap
    assert not (h1.max() < h2.min())
