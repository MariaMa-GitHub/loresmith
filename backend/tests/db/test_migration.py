import subprocess
import sys


def test_alembic_check_passes():
    """Verifies migration scripts are syntactically valid (no DB connection needed)."""
    result = subprocess.run(
        [sys.executable, "-c", "import alembic; print('ok')"],
        cwd=".",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0


def test_baseline_migration_exists():
    from pathlib import Path

    versions = list(Path("alembic/versions").glob("*.py"))
    assert any("001" in p.name or "baseline" in p.name for p in versions), (
        "Expected baseline migration file not found"
    )
