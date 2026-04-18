from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from sqlalchemy import func, select, update

from app.db.models import Passage
from app.db.session import get_session_factory


@dataclass(frozen=True)
class SpoilerReviewEntry:
    passage_id: int
    game_slug: str
    source_url: str
    spoiler_tier: int
    preview: str


def _preview_text(text: str, max_chars: int = 180) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def _parse_override(raw: str) -> tuple[int, int]:
    try:
        passage_id_raw, tier_raw = raw.split("=", maxsplit=1)
        passage_id = int(passage_id_raw)
        spoiler_tier = int(tier_raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid override {raw!r}; expected PASSAGE_ID=TIER"
        ) from exc

    if spoiler_tier < 0 or spoiler_tier > 3:
        raise argparse.ArgumentTypeError(
            f"Invalid spoiler tier {spoiler_tier}; expected 0, 1, 2, or 3"
        )
    return passage_id, spoiler_tier


async def list_passages_for_review(
    *,
    session,
    game_slug: str,
    min_tier: int = 1,
    limit: int = 25,
) -> list[SpoilerReviewEntry]:
    stmt = (
        select(
            Passage.id,
            Passage.game_slug,
            Passage.source_url,
            Passage.spoiler_tier,
            Passage.content,
        )
        .where(Passage.game_slug == game_slug)
        .where(Passage.spoiler_tier >= min_tier)
        .order_by(Passage.spoiler_tier.desc(), Passage.id.asc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    rows = result.all()
    return [
        SpoilerReviewEntry(
            passage_id=row.id,
            game_slug=row.game_slug,
            source_url=row.source_url,
            spoiler_tier=row.spoiler_tier,
            preview=_preview_text(row.content),
        )
        for row in rows
    ]


async def apply_overrides(
    *,
    session,
    game_slug: str,
    overrides: dict[int, int],
) -> int:
    if not overrides:
        return 0

    result = await session.execute(
        select(Passage.id).where(Passage.game_slug == game_slug).where(Passage.id.in_(overrides))
    )
    existing_ids = {row.id for row in result.all()}
    missing = sorted(set(overrides) - existing_ids)
    if missing:
        raise ValueError(
            f"Passages not found for game '{game_slug}': {', '.join(str(pid) for pid in missing)}"
        )

    for passage_id, spoiler_tier in overrides.items():
        await session.execute(
            update(Passage)
            .where(Passage.game_slug == game_slug)
            .where(Passage.id == passage_id)
            .values(
                spoiler_tier=spoiler_tier,
                updated_at=func.now(),
            )
        )
    await session.commit()
    return len(overrides)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect and override spoiler tiers")
    parser.add_argument("--game", required=True, help="Game slug to review.")
    parser.add_argument(
        "--min-tier",
        type=int,
        default=1,
        help="Only show passages at or above this tier.",
    )
    parser.add_argument("--limit", type=int, default=25, help="Max passages to list.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override one passage tier, e.g. --set 123=2. May be repeated.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text output.")
    parser.add_argument("--out", type=Path, help="Optional path to write the review list as JSON.")
    return parser.parse_args()


async def _main() -> None:
    args = _parse_args()
    overrides = dict(_parse_override(item) for item in args.overrides)
    session_factory = get_session_factory()

    async with session_factory() as session:
        if overrides:
            updated = await apply_overrides(
                session=session,
                game_slug=args.game,
                overrides=overrides,
            )
            print(f"Updated spoiler tiers for {updated} passage(s) in {args.game}.")
            return

        entries = await list_passages_for_review(
            session=session,
            game_slug=args.game,
            min_tier=args.min_tier,
            limit=args.limit,
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps([asdict(entry) for entry in entries], indent=2, sort_keys=True)
        )

    if args.json:
        print(json.dumps([asdict(entry) for entry in entries], indent=2, sort_keys=True))
        return

    if not entries:
        print("No passages matched the requested spoiler review filter.")
        return

    for entry in entries:
        print(f"[{entry.passage_id}] tier={entry.spoiler_tier} {entry.source_url}")
        print(f"    {entry.preview}")


if __name__ == "__main__":
    asyncio.run(_main())
