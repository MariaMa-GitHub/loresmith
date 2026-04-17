from app.adapters.base import RobotsPolicy, SourceConfig

_SEED_ARTICLES = [
    "https://hades2.fandom.com/wiki/Melinoe",
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
    "https://hades2.fandom.com/wiki/Skull_(weapon)",
    "https://hades2.fandom.com/wiki/Witch%27s_Staff",
    "https://hades2.fandom.com/wiki/Moonstone_Axe",
    "https://hades2.fandom.com/wiki/Umbral_Flames",
    "https://hades2.fandom.com/wiki/Sister_Blades",
    "https://hades2.fandom.com/wiki/Torc_of_the_Archivist",
    "https://hades2.fandom.com/wiki/Arcana_Cards",
    "https://hades2.fandom.com/wiki/Incantations",
    "https://hades2.fandom.com/wiki/Reagents",
    "https://hades2.fandom.com/wiki/Melinoe%27s_Cauldron",
    "https://hades2.fandom.com/wiki/Crossroads",
    "https://hades2.fandom.com/wiki/Surface_world",
    "https://hades2.fandom.com/wiki/Tartarus",
    "https://hades2.fandom.com/wiki/Oceanus",
    "https://hades2.fandom.com/wiki/Fields_of_Mourning",
    "https://hades2.fandom.com/wiki/Ephyra",
    "https://hades2.fandom.com/wiki/Mount_Olympus",
    "https://hades2.fandom.com/wiki/Prometheus",
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
    chunk_size = 400
    chunk_overlap = 50
    starter_prompts = [
        "Who is Melinoe and how does her story differ from Zagreus's?",
        "What are Arcana Cards and how do they replace the Mirror of Night?",
        "What weapons does Melinoe have access to, and what are their Aspects?",
        "Who is Chronos and why is he the main antagonist?",
        "How does the Surface path differ from the Underworld path in Hades II?",
    ]

    def get_article_urls(self) -> list[str]:
        return list(_SEED_ARTICLES)
