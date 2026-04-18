from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def default_report_path(game_slug: str, reports_dir: Path | None = None) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    base_dir = reports_dir or Path("eval_reports")
    return base_dir / f"{game_slug}-{stamp}.json"


def build_report(
    *,
    run_name: str,
    game_slug: str,
    dataset_path: Path,
    metrics: dict[str, Any],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "run_name": run_name,
        "game_slug": game_slug,
        "dataset_path": str(dataset_path),
        "generated_at": datetime.now(UTC).isoformat(),
        "metrics": metrics,
        "results": results,
    }


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
