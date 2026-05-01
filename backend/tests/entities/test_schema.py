import pytest

from app.entities.schema import (
    ExtractedEntity,
    SchemaValidationError,
    normalize_slug,
    validate_entity,
)


def test_normalize_slug_handles_common_cases():
    assert normalize_slug("Zagreus, Prince of the Underworld") == "zagreus-prince-of-the-underworld"
    assert normalize_slug("  Night Mother  ") == "night-mother"
    assert normalize_slug("Artemis:Boons") == "artemis-boons"


def test_validate_entity_rejects_unknown_type():
    entity = ExtractedEntity(
        slug="zagreus", name="Zagreus",
        entity_type="not-a-real-type", description="",
    )
    with pytest.raises(SchemaValidationError):
        validate_entity(entity, allowed_types={"character", "weapon"})


def test_validate_entity_accepts_known_type():
    entity = ExtractedEntity(
        slug="zagreus", name="Zagreus",
        entity_type="character", description="Prince of the Underworld",
    )
    validate_entity(entity, allowed_types={"character"})  # no exception
