"""
app.py
======
Interactive Streamlit dashboard for the Plane-Wave Expansion Method (PWEM)
photonic-band-diagram calculation.

Refactored from the original ``PWEM.py`` (and its helper module
``convmat.py``). All core numerical work — unit-cell construction, FFT-based
convolution matrices, Bloch-path sampling and the two generalized eigenvalue
problems (E-mode / H-mode) — is preserved exactly as in the original script.
Only the presentation layer changed:

* hardcoded parameters  -> interactive widgets in ``st.sidebar``
* ``print()`` / ``plt.show()`` -> ``st.markdown`` / ``st.metric`` / ``st.pyplot``
* heavy computation     -> ``@st.cache_data`` (recomputed only when inputs change)
* results table + in-memory ``.xlsx`` export (``openpyxl``)
* crystal presets loaded from a local SQLite database (``database.py``)
  — hybrid mode: SQL-driven initial values + free manual sliders

The original script does not read any CSV/Excel file, so no
``st.file_uploader`` is required — the model is fully parameter-driven and
runs immediately on launch.

Run locally with:
    streamlit run app.py
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe backend: Streamlit never calls plt.show()
import matplotlib.pyplot as plt
import streamlit as st
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter
from scipy import linalg

# ``convmat.py`` ships in the same folder as this script (same import as the
# original PWEM.py).
from convmat import convmat

# Local SQLite backend with the crystal preset data (see database.py).
from database import (
    get_crystal_by_name,
    get_crystals,
    init_database,
)


# =============================================================================
# HEAVY COMPUTATION (cached)
# =============================================================================

@st.cache_data(show_spinner="Building unit cell and convolution matrices…")
def build_unit_cell(a: float, r_over_a: float, eps_cyl: float, eps_bg: float,
                    nx: int, p_order: int, q_order: int) -> dict:
    """Build the discretized unit cell and the PWEM convolution matrices.

    Mirrors the original "BUILD UNIT CELL" and "COMPUTE CONVOLUTION MATRIX"
    blocks of PWEM.py one-for-one (the only change: the background
    permittivity, hardcoded to 10 in the original, is now a parameter).
    """
    r = r_over_a * a          # original: r = 0.15 * a

    # --- meshgrid on the cell (original code) -----------------------------
    Nx, Ny = nx, nx           # original: Ny = Nx
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
    ER = eps_bg + (eps_cyl - eps_bg) * ER   # inside: eps_cyl, outside: eps_bg

    # --- convolution matrices (FFT based, see convmat.py) ------------------
    ERC = np.array(convmat(ER, p_order, q_order, Nx, Ny), dtype=complex)
    URC = np.array(convmat(UR, p_order, q_order, Nx, Ny), dtype=complex)

    return {
        "ER": ER,
        "ERC": ERC,
        "URC": URC,
        "xa": xa,
        "ya": ya,
        "r": r,
    }


@st.cache_data(show_spinner="Solving the eigenvalue problem along Γ → X → M → Γ…")
def compute_band_diagram(erc: np.ndarray, urc: np.ndarray,
                         p_order: int, nb: int, a: float) -> dict:
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

        # BUILD EIG-VALUE PROBLEM E-MODE  (A v = λ B v,  A = K·ε⁻¹·K)
        AE = np.dot(np.dot(KX, urc_inv), KX) + np.dot(np.dot(KY, urc_inv), KY)
        BE = erc

        # BUILD EIG-VALUE PROBLEM H-MODE  (A v = λ B v,  A = K·(1/ε)⁻¹·K form)
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

    return {
        "WnE": WnE,
        "WnH": WnH,
        "bx": bx,
        "by": by,
        "n_points": int(np.size(bx)),
    }


# =============================================================================
# FIGURE BUILDERS (thin matplotlib wrappers, same styling as PWEM.py)
# =============================================================================

def make_unit_cell_figure(ER: np.ndarray) -> plt.Figure:
    """Reproduces the original 'SHOW UNIT CELL' plot."""
    fig, ax = plt.subplots(figsize=(4.6, 4.3), dpi=110)
    mesh = ax.pcolormesh(ER)
    fig.colorbar(mesh, ax=ax)
    ax.set_title("UNIT CELL")
    fig.tight_layout()
    return fig


def make_band_figure(WnE: np.ndarray, WnH: np.ndarray, nb: int) -> plt.Figure:
    """Reproduces the two original band-diagram plots (E-mode red, H-mode blue)."""
    upper = 2 * nb + int(np.floor(nb * np.sqrt(2))) - 1
    fig, (ax_e, ax_h) = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=110)

    for ax, wn, color, mode in (
        (ax_e, WnE, "r", "E-mode (TE)"),
        (ax_h, WnH, "b", "H-mode (TM)"),
    ):
        ax.plot(np.real(wn), color)
        ax.set(xlim=[0, upper], ylim=[0, 0.6])
        ax.set_xticks([0, nb, 2 * nb, upper])
        ax.set_xticklabels([r"$\Gamma$", "X", "M", r"$\Gamma$"])
        ax.set_xlabel(r"Bloch wave vector $\vec\beta$")
        ax.set_ylabel(r"Normalized Frequency $\omega$a/2$\pi$c")
        ax.grid(axis="x")
        ax.set_title(f"Band Diagram of a Square Lattice — {mode}")

    fig.tight_layout()
    return fig


# =============================================================================
# EXCEL EXPORT (openpyxl, fully in-memory — no temporary files on disk)
# =============================================================================

@st.cache_data(show_spinner=False)
def build_xlsx_bytes(df: pd.DataFrame) -> bytes:
    """Serialize the results DataFrame to a formatted .xlsx in memory."""
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl", sheet_name="Band diagram")

    # Re-open the workbook to apply openpyxl formatting
    buf.seek(0)
    wb = load_workbook(buf)
    ws = wb.active

    # Bold, centered header row
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal="center")

    # Auto-adjusted column widths (capped so the sheet stays readable)
    for idx, col in enumerate(df.columns, start=1):
        header_len = len(str(col))
        value_lens = [len(str(v)) for v in df[col].astype(str).head(500)]
        width = min(max([header_len, *value_lens]) + 2, 45)
        ws.column_dimensions[get_column_letter(idx)].width = width

    ws.freeze_panes = "A2"  # keep the header visible while scrolling

    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()


# =============================================================================
# UI HELPERS
# =============================================================================

def first_band_edge(wn: np.ndarray) -> float:
    """Smallest positive real frequency over the whole sampled path."""
    pos = np.real(wn)
    pos = pos[pos > 1e-9]
    return float(pos.min()) if pos.size else float("nan")


def build_results_dataframe(wn_e: np.ndarray, wn_h: np.ndarray,
                            bx: np.ndarray, by: np.ndarray,
                            nb: int, n_bands: int) -> pd.DataFrame:
    """Assemble the path table shown in the UI and exported to Excel."""
    n_points = int(np.size(bx))
    n_bands = min(n_bands, wn_e.shape[1])

    segments = (
        ["Γ→X"] * nb + ["X→M"] * nb + ["M→Γ"] * (n_points - 2 * nb)
    )

    data = {
        "Index": np.arange(n_points),
        "Path segment": segments,
        "β_x": np.round(bx, 6),
        "β_y": np.round(by, 6),
    }
    for i in range(n_bands):
        data[f"E band {i + 1}"] = np.round(np.real(wn_e)[:, i], 6)
        data[f"H band {i + 1}"] = np.round(np.real(wn_h)[:, i], 6)

    return pd.DataFrame(data)


# =============================================================================
# PAGE SETUP
# =============================================================================

st.set_page_config(
    page_title="PWEM · Photonic Band Diagrams",
    page_icon="🔬",
    layout="wide",
)

st.title("Plane-Wave Expansion Method (PWEM) — 2-D Photonic Band Diagrams")
st.markdown(
    "This dashboard reproduces the classic **plane-wave expansion** calculation from "
    "`PWEM.py`: a 2-D **square-lattice photonic crystal** with a dielectric cylinder "
    "(radius *r*, permittivity *ε_c*) embedded in a background (permittivity *ε_bg*). "
    "For every Bloch wave vector sampled along the high-symmetry path "
    "**Γ → X → M → Γ**, the generalized eigenvalue problems "
    "$A\\,\\mathbf{v} = \\lambda\\,B\\,\\mathbf{v}$ are solved for the "
    "**E-mode (TE)** and **H-mode (TM)** polarizations, and the normalized "
    "frequencies $\\omega a / 2\\pi c = \\sqrt{\\lambda}$ are plotted as band diagrams."
)

# =============================================================================
# SIDEBAR — all formerly hardcoded inputs live here
# =============================================================================

# Make sure the SQLite database exists and is seeded (first run only).
init_database()

with st.sidebar:
    st.header("⚙️ Simulation parameters")

    # --- Hybrid mode: load initial values from the SQLite database ----------
    st.subheader("Crystal configuration (SQLite)")
    only_high_contrast = st.checkbox(
        "Only high-contrast crystals (rod ε > 10)",
        value=False,
        help="Filters the dropdown list directly in SQL: the query gets an "
             "extra WHERE clause (JOIN + WHERE, secure parameterization).",
    )
    # (The query text/parameters are not displayed in the UI.)
    crystal_rows, _, _ = get_crystals(high_contrast=only_high_contrast)

    if crystal_rows:
        selected_row = st.selectbox(
            "Select a crystal",
            options=crystal_rows,
            format_func=lambda row: (
                f"{row['crystal_name']}  ({row['rod_material']} in "
                f"{row['matrix_material']})"
            ),
            index=0,
        )
        # Single JOINed, parameterized query for the full parameter set.
        crystal = get_crystal_by_name(selected_row["crystal_name"])
    else:
        crystal = None
        st.warning(
            "No crystals match the filter — uncheck it to see the full list."
        )

    # Pre-fill the parameter widgets from the SQL row ("pre-configured
    # widget" pattern): the values are written into the session state ONLY
    # when a *different* crystal is selected. Afterwards the user remains
    # completely free to adjust every slider/field for manual modelling; the
    # values are re-seeded only when the crystal selection actually changes.
    if crystal is not None:
        if st.session_state.get("crystal_id") != crystal["id"]:
            st.session_state.crystal_id = crystal["id"]
            st.session_state["a"] = float(crystal["lattice_const"])
            st.session_state["r_over_a"] = round(
                crystal["rod_radius"] / crystal["lattice_const"], 3
            )
            st.session_state["eps_cyl"] = float(crystal["rod_eps"])
            st.session_state["eps_bg"] = float(crystal["matrix_eps"])
            st.session_state["p_order"] = int(crystal["num_plane_waves"])

    # Fallback defaults for the very first run / empty filter result, so the
    # keyed widgets below always have a session-state value (the widgets must
    # NOT receive a separate `value=` argument when a `key` is used).
    st.session_state.setdefault("a", 2.0)
    st.session_state.setdefault("r_over_a", 0.15)
    st.session_state.setdefault("eps_cyl", 2.0)
    st.session_state.setdefault("eps_bg", 10.0)
    st.session_state.setdefault("p_order", 11)

    st.subheader("Unit cell")
    # The lattice constant has no manual input: it is taken from the selected
    # crystal. In normalized frequency (ωa/2πc) it only sets the absolute
    # scale, so it is displayed in the "Current configuration" panel instead.
    a = st.session_state["a"]
    r_over_a = st.number_input(
        "Cylinder radius r / a",
        min_value=0.05, max_value=0.45, step=0.001,
        key="r_over_a",
        help="Rod radius divided by the lattice constant — auto-loaded from "
             "the selected crystal, free to adjust.",
    )
    eps_cyl = st.number_input(
        "Cylinder permittivity ε_c",
        min_value=0.1, max_value=1000.0, step=0.01,
        key="eps_cyl",
        help="Rod permittivity — type the exact value you need "
             "(auto-loaded from the selected crystal, free to adjust).",
    )
    eps_bg = st.number_input(
        "Background permittivity ε_bg",
        min_value=0.1, max_value=1000.0, step=0.01,
        key="eps_bg",
        help="Matrix permittivity — type the exact value you need "
             "(auto-loaded from the selected crystal, free to adjust).",
    )

    st.subheader("Discretization")
    nx = st.selectbox(
        "Grid resolution N_x = N_y",
        options=[129, 257, 513, 769, 1025],
        index=2,
        help="Odd grid sizes keep the FFT-based convolution matrix centered "
             "(original: Nx = 513). Higher = more accurate but slower.",
    )
    p_order = st.slider(
        "Plane-wave order P (Q = P)",
        min_value=5, max_value=15, step=2,
        key="p_order",
        help=(
            "Number of plane waves per axis in the basis — auto-loaded from "
            "the selected crystal (num_plane_waves). Odd orders keep the "
            "basis symmetric about zero; more basis vectors = better "
            "convergence. Free to adjust."
        ),
    )
    nb = st.number_input(
        "Bloch points per segment N_B",
        min_value=5, max_value=100, value=15, step=1,
        help="Sampling density of the Γ→X and X→M segments; the M→Γ segment "
             "uses floor(N_B·√2) points (original: N_B = 15).",
    )

    st.subheader("Output")
    n_bands_out = st.number_input(
        "Bands shown in table & export",
        min_value=1, max_value=128, value=8, step=1,
        help="How many lowest bands of each mode to display and export.",
    )

    with st.expander("ℹ️ About the model"):
        st.markdown(
            """
**Method.** The fields are expanded as a Fourier series of P×Q plane waves.
For each Bloch vector β, Maxwell's equations reduce to two generalized
eigenvalue problems:

- **E-mode (TE):** `(KX·ε⁻¹·KX + KY·ε⁻¹·KY) v = λ · ε v`
- **H-mode (TM):** `(KX·ε⁻¹·KX + KY·ε⁻¹·KY) v = λ · (1/ε) v`

where `KX`, `KY` are diagonal matrices of the shifted wave vectors
`kx,ky = β − 2π·p/a` and the ε-operators act through the FFT-based
convolution matrices from `convmat.py`.

The normalized frequency is `ωa/2πc = √λ`.

The original script is fully self-contained (no CSV/Excel input), so this
app is parameter-driven — no file upload is needed.
            """
        )

# =============================================================================
# 1 · UNIT CELL
# =============================================================================

st.divider()
st.subheader("1 · Unit cell (relative permittivity)")

cell = build_unit_cell(a, r_over_a, eps_cyl, eps_bg, nx, p_order, p_order)

col_fig, col_info = st.columns([2, 3])
with col_fig:
    fig_cell = make_unit_cell_figure(cell["ER"])
    st.pyplot(fig_cell, use_container_width=True)
    plt.close(fig_cell)  # free the figure memory
    st.caption(
        "Shading: relative permittivity — cylinder "
        f"ε_c = {eps_cyl:g} vs background ε_bg = {eps_bg:g} "
        f"(r = {cell['r']:.3f})."
    )

with col_info:
    st.markdown("**Current configuration**")
    st.markdown(
        f"- Lattice constant: **a = {a:g}**  (cylinder r = {cell['r']:.4f})\n"
        f"- Permittivities: **ε_c = {eps_cyl:g}**, **ε_bg = {eps_bg:g}**\n"
        f"- Grid: **{nx} × {nx}**  ·  Plane-wave basis: **{p_order} × {p_order}** "
        f"({p_order * p_order} vectors)\n"
        f"- Path sampling: **N_B = {nb}** points on Γ→X and X→M, "
        f"**{int(np.floor(nb * np.sqrt(2)))}** on M→Γ"
    )
    st.caption(
        "The model is fully parameter-driven (the original `PWEM.py` reads no "
        "data files), so no `st.file_uploader` is required — everything runs "
        "from the sidebar inputs."
    )

# =============================================================================
# 2 · BAND DIAGRAM
# =============================================================================

st.divider()
st.subheader("2 · Band diagram along Γ → X → M → Γ")

bands = compute_band_diagram(cell["ERC"], cell["URC"], p_order, nb, a)
WnE, WnH = bands["WnE"], bands["WnH"]
bx, by = bands["bx"], bands["by"]

# --- key numerical results (replaces the old print() statements) ------------
e_edge = first_band_edge(WnE)
h_edge = first_band_edge(WnH)
f_max = float(np.max(np.concatenate([np.real(WnE), np.real(WnH)])))
n_solves = 2 * bands["n_points"]

m1, m2, m3, m4 = st.columns(4)
m1.metric("E-mode first band edge (ωa/2πc)", f"{e_edge:.4f}")
m2.metric("H-mode first band edge (ωa/2πc)", f"{h_edge:.4f}")
m3.metric("Max sampled frequency (ωa/2πc)", f"{f_max:.4f}")
m4.metric("Generalized eigenproblems solved", f"{n_solves}")

# --- the two original band plots, side by side --------------------------------
fig_band = make_band_figure(WnE, WnH, nb)
st.pyplot(fig_band, use_container_width=True)
plt.close(fig_band)  # free the figure memory

# =============================================================================
# 3 · DATA TABLE & EXCEL EXPORT
# =============================================================================

st.divider()
st.subheader("3 · Results table & export")

df = build_results_dataframe(WnE, WnH, bx, by, nb, int(n_bands_out))
st.dataframe(df, use_container_width=True, height=420)

# In-memory formatted .xlsx export (no temp files on disk)
xlsx_bytes = build_xlsx_bytes(df)
st.download_button(
    "⬇️ Download band data (.xlsx)",
    data=xlsx_bytes,
    file_name="pwem_band_diagram.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    type="primary",
    help="Exports the table above with bold headers and auto-adjusted columns.",
)

st.caption(
    "Computed with the plane-wave expansion method as in the original `PWEM.py` "
    "and `convmat.py`. Results are cached — changing a parameter recomputes "
    "only what depends on it."
)
