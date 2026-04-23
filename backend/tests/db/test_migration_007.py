"""Structural smoke test for migration 007."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory

BACKEND_ROOT = Path(__file__).resolve().parents[2]


def test_migration_007_is_discoverable():
    cfg = Config(str(BACKEND_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_ROOT / "alembic"))
    scripts = ScriptDirectory.from_config(cfg)
    revisions = {s.revision for s in scripts.walk_revisions()}
    assert "007" in revisions
    script = scripts.get_revision("007")
    assert script is not None
    assert script.down_revision == "006"


def test_migration_007_module_exposes_upgrade_and_downgrade():
    path = BACKEND_ROOT / "alembic" / "versions" / "007_semantic_cache_scope_columns.py"
    spec = spec_from_file_location("migration_007", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.upgrade)
    assert callable(module.downgrade)
