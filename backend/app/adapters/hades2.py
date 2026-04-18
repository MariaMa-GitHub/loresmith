from app.adapters.base import (
    EntityTypeSchema,
    RobotsPolicy,
    SourceConfig,
    SpoilerProfile,
)
from app.ingestion.chunker import Chunker

_SEED_ARTICLES = [
    "https://hades2.fandom.com/wiki/Melino%C3%AB",
    "https://hades2.fandom.com/wiki/Chronos",
    "https://hades2.fandom.com/wiki/Hecate",
    "https://hades2.fandom.com/wiki/Charon",
    "https://hades2.fandom.com/wiki/Nemesis",
    "https://hades2.fandom.com/wiki/Moros",
    "https://hades2.fandom.com/wiki/Hypnos",
    "https://hades2.fandom.com/wiki/Chaos",
    "https://hades2.fandom.com/wiki/Zeus",
    "https://hades2.fandom.com/wiki/Poseidon",
    "https://hades2.fandom.com/wiki/Aphrodite",
    "https://hades2.fandom.com/wiki/Hephaestus",
    "https://hades2.fandom.com/wiki/Apollo",
    "https://hades2.fandom.com/wiki/Hestia",
    "https://hades2.fandom.com/wiki/Selene",
    "https://hades2.fandom.com/wiki/Hermes",
    "https://hades2.fandom.com/wiki/Demeter",
    "https://hades2.fandom.com/wiki/Argent_Skull",
    "https://hades2.fandom.com/wiki/Witch%27s_Staff",
    "https://hades2.fandom.com/wiki/Moonstone_Axe",
    "https://hades2.fandom.com/wiki/Umbral_Flames",
    "https://hades2.fandom.com/wiki/Sister_Blades",
    "https://hades2.fandom.com/wiki/Keepsakes/Hades_II",
    "https://hades2.fandom.com/wiki/Arcana_Cards",
    "https://hades2.fandom.com/wiki/Incantations",
    "https://hades2.fandom.com/wiki/Reagents",
    "https://hades2.fandom.com/wiki/Cauldron",
    "https://hades2.fandom.com/wiki/Crossroads",
    "https://hades2.fandom.com/wiki/The_Surface",
    "https://hades2.fandom.com/wiki/Tartarus",
    "https://hades2.fandom.com/wiki/Oceanus",
    "https://hades2.fandom.com/wiki/Fields_of_Mourning",
    "https://hades2.fandom.com/wiki/City_of_Ephyra",
    "https://hades2.fandom.com/wiki/Mount_Olympus",
    "https://hades2.fandom.com/wiki/Prometheus",
]


_HADES2_SPOILER_PROFILE = SpoilerProfile(
    tier_keywords=(
        (3, (
            "true ending", "final ending", "epilogue", "postgame",
        )),
        (2, (
            "after defeating chronos", "after chronos is defeated",
            "rescue hades", "hades is rescued", "ending reveal",
        )),
        (1, (
            "chronos", "titan of time", "surface route", "surface world",
            "mount olympus", "olympus route", "ephyra",
        )),
    ),
    ambiguous_keywords=(
        "chronos", "surface", "olympus", "ending", "titan", "reveal",
    ),
    system_prompt="""You are a spoiler classifier for the video game Hades II.
Assign a spoiler tier to the passage:
0 = safe (trailers, early Crossroads characters, starter weapons, broad mechanics)
1 = moderate (Chronos as antagonist, later-route biomes, notable progression reveals)
2 = major (late-Early-Access story beats, post-boss revelations, ending specifics)
3 = endgame (explicit final ending or postgame resolution)

Be conservative: if unsure, prefer the higher tier.
Reply with a single digit: 0, 1, 2, or 3. Nothing else.""",
    fallback_ambiguous_tier=2,
)


_HADES2_ENTITY_SCHEMA = [
    EntityTypeSchema(
        name="character",
        description="Named person, god, or shade in the Hades II world.",
    ),
    EntityTypeSchema(name="weapon", description="Nocturnal Arm or weapon aspect."),
    EntityTypeSchema(name="region", description="Underworld biome, Surface locale, or encampment."),
    EntityTypeSchema(name="boon", description="Olympian-granted in-run power-up."),
    EntityTypeSchema(name="arcana_card", description="Arcana Card providing between-run passives."),
    EntityTypeSchema(
        name="incantation",
        description="Reagent-based incantation from Melinoe's Cauldron.",
    ),
    EntityTypeSchema(name="resource", description="Reagent, currency, or consumable."),
    EntityTypeSchema(
        name="mechanic",
        description="Persistent system such as Incantations or Arcana.",
    ),
]


class HadesIIAdapter:
    slug = "hades2"
    display_name = "Hades II"
    sources = [
        SourceConfig(
            base_url="https://hades2.fandom.com",
            allowed_path_prefix="/wiki/",
            license="CC-BY-SA-3.0",
            crawl_delay=1.5,
        )
    ]
    robots_policy = RobotsPolicy.RESPECT
    license = "CC-BY-SA-3.0"
    chunker = Chunker(chunk_size=400, overlap=50)
    starter_prompts = [
        "Who is Melinoe and how does her story differ from Zagreus's?",
        "What are Arcana Cards and how do they replace the Mirror of Night?",
        "What weapons does Melinoe have access to, and what are their Aspects?",
        "Who is Chronos and why is he the main antagonist?",
        "How does the Surface path differ from the Underworld path in Hades II?",
    ]
    spoiler_profile = _HADES2_SPOILER_PROFILE
    entity_schema = _HADES2_ENTITY_SCHEMA

    def get_article_urls(self) -> list[str]:
        return list(_SEED_ARTICLES)
