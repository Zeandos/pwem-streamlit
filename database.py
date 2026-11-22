"""
database.py
===========
SQLite relational backend for the PWEM Streamlit dashboard (``app.py``).

Schema (normalized, two related tables)
---------------------------------------
``materials``  (reference table: materials and their dielectric constants)
    id   INTEGER PRIMARY KEY AUTOINCREMENT
    name TEXT    NOT NULL UNIQUE
    eps  REAL    NOT NULL          -- relative dielectric constant

``crystals``   (photonic crystal configurations)
    id                 INTEGER PRIMARY KEY AUTOINCREMENT
    name               TEXT    NOT NULL UNIQUE
    lattice_const      REAL    NOT NULL    -- lattice constant a
    rod_radius         REAL    NOT NULL    -- rod radius r (absolute)
    rod_material_id    INTEGER NOT NULL    -- FOREIGN KEY -> materials.id
    matrix_material_id INTEGER NOT NULL    -- FOREIGN KEY -> materials.id
    num_plane_waves    INTEGER NOT NULL    -- plane-wave basis order P

On the first run (empty database) the schema is created and both tables are
populated with reference data; later runs leave existing data untouched
(idempotent).

Conventions
-----------
* ``sqlite3.Row`` is used as the ``row_factory`` so every fetched row behaves
  like a dictionary (e.g. ``row["eps"]``).
* Every query uses secure parameterization (``?`` placeholders) — no string
  interpolation of user data.
"""

from __future__ import annotations

import os
import sqlite3

# =============================================================================
# Constants
# =============================================================================

#: The database file lives next to this module (no hardcoded absolute path).
DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "pwem_crystals.db"
)

#: A crystal is "high contrast" when the rod dielectric constant exceeds this
#: threshold (the sidebar filter adds ``WHERE rod.eps > ?`` with this value).
HIGH_CONTRAST_EPS_THRESHOLD = 10.0

# --- reference dielectric constants (relative permittivity) -----------------
SEED_MATERIALS = [
    ("Silicon", 11.7),
    ("Air", 1.0),
    ("Gallium Arsenide", 12.1),
    ("Titanium Dioxide", 8.4),
    ("Water", 1.77),
    ("SU-8 polymer", 4.0),
]

# --- five photonic crystal configurations ------------------------------------
# (name, lattice_const, rod_radius, rod_material, matrix_material, num_plane_waves)
SEED_CRYSTALS = [
    ("Silicon rods in Air", 1.55, 0.23, "Silicon", "Air", 11),
    ("GaAs rods in Water", 2.00, 0.30, "Gallium Arsenide", "Water", 9),
    ("TiO2 rods in SU-8", 4.00, 0.70, "Titanium Dioxide", "SU-8 polymer", 11),
    ("Air holes in Silicon", 5.43, 0.90, "Air", "Silicon", 13),
    ("SU-8 rods in Water", 3.00, 0.50, "SU-8 polymer", "Water", 7),
]

# --- SQL statements (single source of truth, also displayed in the UI) ------

#: Base query for the crystal list: one JOIN per materials relation so each
#: row already carries every parameter the UI needs.
CRYSTALS_LIST_SQL = """
    SELECT
        c.id,
        c.name            AS crystal_name,
        c.lattice_const,
        c.rod_radius,
        c.num_plane_waves,
        rod.name          AS rod_material,
        rod.eps           AS rod_eps,
        mat.name          AS matrix_material,
        mat.eps           AS matrix_eps
    FROM crystals AS c
    JOIN materials AS rod ON rod.id = c.rod_material_id
    JOIN materials AS mat ON mat.id = c.matrix_material_id
"""

#: WHERE clause used for the "high contrast" filter (sidebar checkbox).
HIGH_CONTRAST_WHERE_CLAUSE = "WHERE rod.eps > ?"

#: Single-crystal lookup: one JOINed, parameterized query.
CRYSTAL_BY_NAME_SQL = """
    SELECT
        c.id,
        c.name            AS crystal_name,
        c.lattice_const,
        c.rod_radius,
        c.num_plane_waves,
        rod.name          AS rod_material,
        rod.eps           AS rod_eps,
        mat.name          AS matrix_material,
        mat.eps           AS matrix_eps
    FROM crystals AS c
    JOIN materials AS rod ON rod.id = c.rod_material_id
    JOIN materials AS mat ON mat.id = c.matrix_material_id
    WHERE c.name = ?
"""


# =============================================================================
# Connection / schema
# =============================================================================

def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Open a connection with dict-like rows (``sqlite3.Row``) and active
    foreign-key enforcement."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_database(db_path: str = DB_PATH) -> str:
    """Create the schema on the first run (empty database) and seed data.

    Idempotent: if tables already contain data they are left untouched.
    Returns the path of the database file.
    """
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS materials (
                    id   INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    eps  REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS crystals (
                    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                    name               TEXT NOT NULL UNIQUE,
                    lattice_const      REAL NOT NULL,
                    rod_radius         REAL NOT NULL,
                    rod_material_id    INTEGER NOT NULL
                                       REFERENCES materials (id),
                    matrix_material_id INTEGER NOT NULL
                                       REFERENCES materials (id),
                    num_plane_waves    INTEGER NOT NULL
                                       CHECK (num_plane_waves > 0
                                              AND num_plane_waves % 2 = 1)
                )
                """
            )

        # --- populate only when the database is empty (first run) -----------
        n_materials = conn.execute(
            "SELECT COUNT(*) AS n FROM materials"
        ).fetchone()["n"]
        if n_materials == 0:
            with conn:
                conn.executemany(
                    "INSERT INTO materials (name, eps) VALUES (?, ?)",
                    SEED_MATERIALS,
                )

        n_crystals = conn.execute(
            "SELECT COUNT(*) AS n FROM crystals"
        ).fetchone()["n"]
        if n_crystals == 0:
            material_ids = {
                row["name"]: row["id"]
                for row in conn.execute("SELECT id, name FROM materials")
            }
            rows = [
                (name, lattice_const, rod_radius,
                 material_ids[rod_material], material_ids[matrix_material],
                 num_plane_waves)
                for (name, lattice_const, rod_radius,
                     rod_material, matrix_material, num_plane_waves)
                in SEED_CRYSTALS
            ]
            with conn:
                conn.executemany(
                    """
                    INSERT INTO crystals
                        (name, lattice_const, rod_radius,
                         rod_material_id, matrix_material_id,
                         num_plane_waves)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
    finally:
        conn.close()
    return db_path


# =============================================================================
# Queries
# =============================================================================

def get_crystals(high_contrast: bool = False):
    """Retrieve the list of crystals (JOINing both materials relations).

    With ``high_contrast=True`` an extra ``WHERE rod.eps > ?`` clause is
    appended (secure parameterization — the threshold is a bound parameter,
    never interpolated into the SQL text).

    Returns
    -------
    (rows, sql, params)
        ``rows``    list of dict-like mappings (one per crystal)
        ``sql``     the exact SQL statement that was executed
        ``params``  the bound parameters (``()`` when no WHERE clause)
    """
    sql = CRYSTALS_LIST_SQL
    params: list = []
    if high_contrast:
        sql += f"\n    {HIGH_CONTRAST_WHERE_CLAUSE}\n"
        params.append(HIGH_CONTRAST_EPS_THRESHOLD)
    sql += "    ORDER BY c.name\n"

    conn = get_connection()
    try:
        rows = [dict(row) for row in conn.execute(sql, params)]
    finally:
        conn.close()
    return rows, sql, params


def get_crystal_by_name(name: str):
    """Retrieve ALL parameters of one crystal with a single SQL query.

    One JOINed statement (crystals -> materials x2) with secure
    parameterization (``?``). Returns a dict-like mapping, or ``None`` when
    the crystal name is unknown.
    """
    conn = get_connection()
    try:
        row = conn.execute(CRYSTAL_BY_NAME_SQL, (name,)).fetchone()
    finally:
        conn.close()
    return dict(row) if row is not None else None


# =============================================================================
# Small CLI for a quick manual check:  python database.py
# =============================================================================

if __name__ == "__main__":
    path = init_database()
    print(f"Database: {path}")
    rows, sql, params = get_crystals(high_contrast=False)
    print(f"\nAll crystals ({len(rows)}):")
    for r in rows:
        print(
            f"  - {r['crystal_name']:<22} a={r['lattice_const']:>5.2f}  "
            f"r={r['rod_radius']:>4.2f}  P={r['num_plane_waves']:>2}  "
            f"rod {r['rod_eps']:>5.2f} / matrix {r['matrix_eps']:>5.2f}"
        )
    hc, _, _ = get_crystals(high_contrast=True)
    print(f"\nHigh-contrast (rod eps > {HIGH_CONTRAST_EPS_THRESHOLD:g}): "
          f"{[r['crystal_name'] for r in hc]}")
