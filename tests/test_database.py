"""
test_database.py — unit tests for the SQLite data access layer (src/database).

Each test uses a throw-away database in pytest's ``tmp_path`` — the project's
real ``data/pwem_crystals.db`` is never touched.
"""

from __future__ import annotations

import sqlite3

import pytest

from src import database as db


@pytest.fixture()
def db_path(tmp_path):
    """A fresh, initialized database inside the temporary test directory."""
    path = str(tmp_path / "test_crystals.db")
    db.init_database(path)
    return path


# --- schema & seeding ---------------------------------------------------------


def test_init_creates_tables_and_seeds(db_path):
    conn = db.get_connection(db_path)
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    conn.close()
    assert {"materials", "crystals"} <= tables


def test_seed_counts(db_path):
    conn = db.get_connection(db_path)
    n_materials = conn.execute(
        "SELECT COUNT(*) FROM materials"
    ).fetchone()[0]
    n_crystals = conn.execute(
        "SELECT COUNT(*) FROM crystals"
    ).fetchone()[0]
    conn.close()
    assert n_materials == len(db.SEED_MATERIALS) == 6
    assert n_crystals == len(db.SEED_CRYSTALS) == 5


def test_init_is_idempotent(db_path):
    db.init_database(db_path)  # second run must not duplicate the data
    rows, _, _ = db.get_crystals(db_path=db_path)
    assert len(rows) == 5


# --- connection / integrity ---------------------------------------------------


def test_foreign_keys_enforced(db_path):
    conn = db.get_connection(db_path)
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    conn.close()


def test_odd_plane_waves_check_constraint(db_path):
    conn = db.get_connection(db_path)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO crystals (name, lattice_const, rod_radius,"
            " rod_material_id, matrix_material_id, num_plane_waves)"
            " VALUES ('even P', 1.0, 0.1, 1, 2, 8)"
        )
    conn.close()


# --- queries ------------------------------------------------------------------


def test_get_crystals_returns_all(db_path):
    rows, sql, params = db.get_crystals(db_path=db_path)
    assert len(rows) == 5
    assert params == []  # no WHERE clause -> no bound parameters
    # every row carries both materials relations already JOINed
    row = rows[0]
    for key in (
        "crystal_name", "lattice_const", "rod_radius", "num_plane_waves",
        "rod_material", "rod_eps", "matrix_material", "matrix_eps",
    ):
        assert key in row


def test_high_contrast_filter(db_path):
    rows, sql, params = db.get_crystals(high_contrast=True, db_path=db_path)
    # secure parameterization: threshold is bound, not interpolated
    assert params == [db.HIGH_CONTRAST_EPS_THRESHOLD]
    assert "WHERE rod.eps > ?" in sql
    names = {row["crystal_name"] for row in rows}
    assert names == {"Silicon rods in Air", "GaAs rods in Water"}
    assert all(row["rod_eps"] > 10.0 for row in rows)


def test_get_crystal_by_name(db_path):
    crystal = db.get_crystal_by_name("Silicon rods in Air", db_path=db_path)
    assert crystal is not None
    assert crystal["lattice_const"] == pytest.approx(1.55)
    assert crystal["rod_radius"] == pytest.approx(0.23)
    assert crystal["rod_eps"] == pytest.approx(11.7)
    assert crystal["matrix_eps"] == pytest.approx(1.0)
    assert crystal["num_plane_waves"] == 11


def test_get_crystal_by_name_unknown(db_path):
    assert db.get_crystal_by_name("NoSuchCrystal", db_path=db_path) is None
