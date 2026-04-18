from app.adapters.base import (
    EntityTypeSchema,
    RobotsPolicy,
    SourceConfig,
    SpoilerProfile,
)
from app.ingestion.chunker import Chunker

# Static seed list of Hades Wiki pages to scrape.
# All pages are CC-BY-SA-3.0: https://hades.fandom.com/wiki/
# Sitemap-based auto-discovery is deferred to a future enhancement.
# Note: Apollo only appears in Hades II, but the wiki page exists and contains
# relevant cross-game lore. Content is spoiler-tagged appropriately.
_SEED_ARTICLES = [
    # --- Characters: protagonist + House of Hades ---
    "https://hades.fandom.com/wiki/Zagreus",
    "https://hades.fandom.com/wiki/Hades",
    "https://hades.fandom.com/wiki/Nyx",
    "https://hades.fandom.com/wiki/Persephone",
    "https://hades.fandom.com/wiki/Cerberus",
    "https://hades.fandom.com/wiki/Achilles",
    "https://hades.fandom.com/wiki/Hypnos",
    "https://hades.fandom.com/wiki/Orpheus",
    "https://hades.fandom.com/wiki/Dusa",
    "https://hades.fandom.com/wiki/Skelly",
    # --- Characters: Tartarus bosses ---
    "https://hades.fandom.com/wiki/Megaera",
    "https://hades.fandom.com/wiki/Alecto",
    "https://hades.fandom.com/wiki/Tisiphone",
    # --- Characters: Asphodel boss ---
    "https://hades.fandom.com/wiki/Bone_Hydra",
    # --- Characters: Elysium bosses ---
    "https://hades.fandom.com/wiki/Theseus",
    "https://hades.fandom.com/wiki/Asterius",
    # --- Characters: mid-run encounters ---
    "https://hades.fandom.com/wiki/Patroclus",
    "https://hades.fandom.com/wiki/Sisyphus",
    "https://hades.fandom.com/wiki/Eurydice",
    "https://hades.fandom.com/wiki/Charon",
    "https://hades.fandom.com/wiki/Thanatos",
    "https://hades.fandom.com/wiki/Chaos",
    # --- Olympian gods ---
    "https://hades.fandom.com/wiki/Olympians",
    "https://hades.fandom.com/wiki/Zeus",
    "https://hades.fandom.com/wiki/Poseidon",
    "https://hades.fandom.com/wiki/Athena",
    "https://hades.fandom.com/wiki/Aphrodite",
    "https://hades.fandom.com/wiki/Ares",
    "https://hades.fandom.com/wiki/Artemis",
    "https://hades.fandom.com/wiki/Apollo",
    "https://hades.fandom.com/wiki/Dionysus",
    "https://hades.fandom.com/wiki/Demeter",
    "https://hades.fandom.com/wiki/Hermes",
    # --- Weapons (Infernal Arms) ---
    "https://hades.fandom.com/wiki/Stygian_Blade",
    "https://hades.fandom.com/wiki/Heart-Seeking_Bow",
    "https://hades.fandom.com/wiki/Shield_of_Chaos",
    "https://hades.fandom.com/wiki/Eternal_Spear",
    "https://hades.fandom.com/wiki/Twin_Fists",
    "https://hades.fandom.com/wiki/Adamant_Rail",
    "https://hades.fandom.com/wiki/Daedalus_Hammer",
    # --- Regions ---
    "https://hades.fandom.com/wiki/Tartarus",
    "https://hades.fandom.com/wiki/Asphodel",
    "https://hades.fandom.com/wiki/Elysium",
    "https://hades.fandom.com/wiki/Temple_of_Styx",
    "https://hades.fandom.com/wiki/House_of_Hades",
    # --- Mechanics ---
    "https://hades.fandom.com/wiki/Mirror_of_Night",
    "https://hades.fandom.com/wiki/Pact_of_Punishment",
    "https://hades.fandom.com/wiki/Boons",
    "https://hades.fandom.com/wiki/Keepsakes",
    "https://hades.fandom.com/wiki/Codex",
    "https://hades.fandom.com/wiki/Companions",
    "https://hades.fandom.com/wiki/Fated_List_of_Minor_Prophecies",
    "https://hades.fandom.com/wiki/Wretched_Broker",
    "https://hades.fandom.com/wiki/Well_of_Charon",
    # --- Resources / currencies ---
    "https://hades.fandom.com/wiki/Charon%27s_Obol",
    "https://hades.fandom.com/wiki/Darkness",
    "https://hades.fandom.com/wiki/Chthonic_Key",
    "https://hades.fandom.com/wiki/Gemstones",
    "https://hades.fandom.com/wiki/Nectar",
    "https://hades.fandom.com/wiki/Ambrosia",
    "https://hades.fandom.com/wiki/Titan_Blood",
    "https://hades.fandom.com/wiki/Diamond",
]


_HADES_SPOILER_PROFILE = SpoilerProfile(
    tier_keywords=(
        (3, (
            "ten escapes", "10 escapes", "true ending", "final ending",
            "reconcile", "reconciliation", "persephone returns",
            "hades forgives", "epilogue",
        )),
        (2, (
            "persephone is zagreus", "zagreus's mother", "zagreus's true mother",
            "true mother", "cannot survive", "dies on the surface",
            "real mother", "mother's identity",
        )),
        (1, (
            "first escape", "reaches the surface", "reaches the mortal world",
            "achilles and patroclus reunite", "patroclus reunite",
            "sisyphus's punishment lightened",
        )),
    ),
    ambiguous_keywords=(
        "origins", "mother", "family", "secret", "reveals", "truth",
        "surface", "escape", "ending",
    ),
    system_prompt="""You are a spoiler classifier for the video game Hades.
Assign a spoiler tier to the passage:
0 = safe (mechanics, early characters, no plot reveals)
1 = minor (mid-game character arcs, first escape)
2 = major (Persephone identity, surface survival reveal)
3 = endgame (true ending, post-10-escape story)

Reply with a single digit: 0, 1, 2, or 3. Nothing else.""",
    fallback_ambiguous_tier=2,
)


_HADES_ENTITY_SCHEMA = [
    EntityTypeSchema(
        name="character",
        description="Named person, god, or shade in the Underworld.",
    ),
    EntityTypeSchema(name="weapon", description="Infernal Arm or weapon aspect."),
    EntityTypeSchema(name="region", description="Underworld biome or location."),
    EntityTypeSchema(name="boon", description="Olympian-granted in-run power-up."),
    EntityTypeSchema(name="keepsake", description="Gift-unlocked passive equipped before a run."),
    EntityTypeSchema(name="resource", description="Currency, material, or consumable."),
    EntityTypeSchema(
        name="mechanic",
        description="Persistent system such as Mirror of Night or Pact of Punishment.",
    ),
]


class HadesAdapter:
    slug = "hades"
    display_name = "Hades"
    sources = [
        SourceConfig(
            base_url="https://hades.fandom.com",
            allowed_path_prefix="/wiki/",
            license="CC-BY-SA-3.0",
            crawl_delay=1.5,
        )
    ]
    robots_policy = RobotsPolicy.RESPECT
    license = "CC-BY-SA-3.0"
    chunker = Chunker(chunk_size=400, overlap=50)
    starter_prompts = [
        "Who is Zagreus and why is he trying to escape the Underworld?",
        "What are the six Infernal Arms and how do I unlock them?",
        "How do boon synergies and Duo Boons work?",
        "What is the Pact of Punishment and which conditions are most impactful?",
        "What happens narratively when you first escape the Underworld?",
    ]
    spoiler_profile = _HADES_SPOILER_PROFILE
    entity_schema = _HADES_ENTITY_SCHEMA

    def get_article_urls(self) -> list[str]:
        return list(_SEED_ARTICLES)
