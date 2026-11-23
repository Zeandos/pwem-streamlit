"""
config.py
=========
Central settings for the PWEM project: file paths, environment-variable
overrides and default simulation parameters.

Every other module (``src/*``, ``app.py``, ``tests/*``) reads its constants
from here — no hardcoded paths or magic numbers elsewhere.

``.env``
--------
A small ``.env`` file next to this module is loaded automatically
(KEY=VALUE lines, ``#`` comments) so the database location can be overridden
without code changes, e.g.::

    PWEM_DB_PATH=/absolute/path/to/other.db
"""

from __future__ import annotations

import os
from pathlib import Path

# =============================================================================
# Paths
# =============================================================================

#: Project root (the directory containing this file).
BASE_DIR: Path = Path(__file__).resolve().parent

#: Local data storage directory (created on demand).
DATA_DIR: Path = BASE_DIR / "data"


def _load_dotenv() -> None:
    """Load KEY=VALUE pairs from ``.env`` into ``os.environ`` (no override of
    variables already set). Minimal parser — no external dependency."""
    env_file = BASE_DIR / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


_load_dotenv()


def _default_db_path() -> Path:
    return DATA_DIR / "pwem_crystals.db"


#: SQLite database file (``PWEM_DB_PATH`` env var wins, else ``data/*.db``).
DB_PATH: str = os.environ.get("PWEM_DB_PATH") or str(_default_db_path())

# =============================================================================
# Default simulation parameters (sidebar fallbacks)
# =============================================================================

#: Allowed grid sizes for the unit-cell discretization (odd sizes keep the
#: FFT-based convolution matrix centered).
GRID_OPTIONS: list[int] = [129, 257, 513, 769, 1025]

#: Default grid size (index 2 of GRID_OPTIONS).
DEFAULT_NX: int = 513

#: Default plane-wave order P (odd, Q = P).
DEFAULT_P_ORDER: int = 11

#: Default number of Bloch points per Γ→X and X→M segment.
DEFAULT_NB: int = 15

#: Default number of bands shown in the table / Excel export.
DEFAULT_N_BANDS: int = 8

#: Fallback crystal parameters when the database yields no row.
DEFAULT_A: float = 2.0
DEFAULT_R_OVER_A: float = 0.15
DEFAULT_EPS_CYL: float = 2.0
DEFAULT_EPS_BG: float = 10.0
