# PWEM — 2-D Photonic Band Diagrams (Streamlit)

Interactive Streamlit dashboard for computing photonic band diagrams with the
**Plane-Wave Expansion Method (PWEM)** — a production-ready refactoring of the
original `PWEM.py` script (kept untouched in this repository for reference):
identical numerics, a modern UI, SQLite-based crystal presets and one-click
Excel export.

**The model.** A 2-D photonic crystal: a dielectric cylinder (permittivity
ε_c, radius r) in a background matrix (permittivity ε_bg) on a square lattice
with lattice constant `a`. For each Bloch vector β along the high-symmetry
path Γ → X → M → Γ, Maxwell's equations reduce to two generalized eigenvalue
problems (E-mode / TE and H-mode / TM), solved in a P×Q plane-wave basis with
FFT-based convolution matrices.

---

## Quick start

```bash
git clone <repo-url>
cd PWEM_strimlink
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # optional — defaults work out of the box
streamlit run app.py               # → http://localhost:8501
```

That's it. The SQLite database (`data/pwem_crystals.db`) and its schema are
**created and seeded automatically on first run** — there is no database file
in this repository, by design.

Requirements: Python 3.10+ (tested with 3.13); dependencies:
`streamlit`, `numpy`, `scipy`, `pandas`, `matplotlib`, `openpyxl`
(+ `pytest` for development).

---

## Project structure

```
PWEM_strimlink/
├── app.py                  # Entry point (Streamlit UI) — rendering & widgets only
├── config.py               # Settings, constants, file paths, environment variables
├── requirements.txt        # Project dependencies
├── .env.example            # Configuration template (committed) — copy to .env
├── .gitignore              # Excludes *.db, __pycache__, .env, .venv, …
├── conftest.py             # Makes `src`/`config` importable under pytest
├── PWEM.py                 # Original script, kept untouched for reference
├── src/                    # Core source package (no Streamlit imports here)
│   ├── __init__.py
│   ├── physics.py          # Business/domain logic (PWEM calculations)
│   ├── convmat.py          # FFT-based convolution-matrix helper (from the original)
│   ├── database.py         # Data access layer (SQLite: schema, seeds, queries)
│   └── models.py           # Data structures (dataclasses: UnitCell, BandDiagram)
├── data/                   # Local data storage (git-ignored, auto-created)
│   └── pwem_crystals.db    # SQLite database file
└── tests/                  # Automated unit and integration tests
    ├── test_database.py
    └── test_physics.py
```

### Where to change what

| Task | File |
| --- | --- |
| Widgets, plots, table, Excel export, layout | `app.py` |
| The math (geometry, eigenproblems, path sampling) | `src/physics.py` |
| New materials / crystal presets | `src/database.py` → `SEED_MATERIALS` / `SEED_CRYSTALS` (then delete `data/*.db` once to re-seed) |
| Defaults, grid options, DB path, thresholds | `config.py` |
| Database location / secrets on this machine | `.env` (copy of `.env.example`, git-ignored) |
| Data structures shared by physics and UI | `src/models.py` |
| Regression tests for any change above | `tests/test_database.py`, `tests/test_physics.py` |

Dependency direction is strict: `app.py → src.* → config`; `src/physics.py`
has **no** Streamlit imports, so it stays unit-testable.

---

## Configuration

All settings live in `config.py` (single source of truth). Two ways to
override them locally:

1. **Environment variable** — e.g. `PWEM_DB_PATH=/path/to/other.db`
2. **`.env` file** (project root) — a minimal parser in `config.py` reads
   `KEY=VALUE` lines; no extra dependency.

```bash
cp .env.example .env    # then edit .env
```

> **Security policy.** `.env` is git-ignored and **never committed** — it may
> contain local paths or credentials. The repository ships only
> `.env.example`, which lists the configuration keys as placeholders.
> Likewise, **database files are never committed** (see `.gitignore`); the
> database is generated and re-seeded automatically at runtime.

---

## Using the app

| Sidebar section | Controls |
| --- | --- |
| **Crystal configuration (SQLite)** | Select a preset crystal; optional "high-contrast only" filter (rod ε > 10) |
| **Unit cell** | Cylinder radius r/a, cylinder permittivity ε_c, background permittivity ε_bg (exact numeric entry) |
| **Discretization** | Grid resolution N_x = N_y (129–1025, default 513), plane-wave order P (slider, Q = P), Bloch points per segment N_B |
| **Output** | Number of bands shown in the table & export |

- Selecting a crystal loads `a`, r/a, both permittivities and P from the
  database; every field then stays freely editable (hybrid preset/manual
  mode). Switching crystals re-loads the presets.
- Main page, in order: **1** unit-cell map, **2** band diagram with summary
  metrics (first-band edges, max frequency, eigenproblems solved), **3**
  results table + Excel download button (formatted in memory, no temp files).
- Heavy computation is wrapped in `@st.cache_data` — changing a parameter
  recomputes only what depends on it.

## Data model

`src/database.py` (the data access layer) manages two related tables —
`PRAGMA foreign_keys = ON`, parameterized `?` queries only. The database file
lives in `data/` (path resolved by `config.py`):

```
materials(id, name, eps)  ──<  crystals(id, name, lattice_const, rod_radius,
                                 num_plane_waves, rod_material_id, matrix_material_id)
```

Seeded data — **materials** (name, ε): Silicon 11.7 · Air 1.0 · Gallium
Arsenide 12.1 · Titanium Dioxide 8.4 · Water 1.77 · SU-8 polymer 4.0.
**Crystal presets** (name, a, r, P):

| Crystal | a | r | P | Rod / matrix |
| --- | --- | --- | --- | --- |
| Silicon rods in Air | 1.55 | 0.23 | 11 | Silicon / Air |
| GaAs rods in Water | 2.00 | 0.30 | 9 | Gallium Arsenide / Water |
| TiO2 rods in SU-8 | 4.00 | 0.70 | 11 | Titanium Dioxide / SU-8 |
| Air holes in Silicon | 5.43 | 0.90 | 13 | Air / Silicon |
| SU-8 rods in Water | 3.00 | 0.50 | 7 | SU-8 / Water |

Delete `data/pwem_crystals.db` to start fresh — it is re-created idempotently
(seeding runs only on an empty database, so your manual edits are never lost).

## Testing

```bash
python -m pytest            # or: pytest
```

15 tests covering the SQLite layer (schema, seeding, idempotency, FK/CHECK
integrity, parameterized queries — all against throw-away databases in
`tmp_path`) and the physics engine (material-mask regression, convolution
matrix properties, band ordering, and the TE band gap of high-contrast rods).

**Workflow:** edit → `python -m pytest` → `streamlit run app.py` → commit.
