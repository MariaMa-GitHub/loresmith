"use client";

const SPOILER_TIERS = [
  {
    value: 0,
    label: "Safe",
    description: "Mechanics, early lore, and low-spoiler answers only.",
  },
  {
    value: 1,
    label: "Minor",
    description: "Allows mid-run reveals and lighter character-arc details.",
  },
  {
    value: 2,
    label: "Major",
    description: "Allows late-story revelations and heavier plot details.",
  },
  {
    value: 3,
    label: "Endgame",
    description: "Allows full endgame and epilogue-level spoilers.",
  },
] as const;

interface SpoilerSliderProps {
  value: number;
  onChange: (value: number) => void;
}

export function SpoilerSlider({ value, onChange }: SpoilerSliderProps) {
  const activeTier = SPOILER_TIERS[value] ?? SPOILER_TIERS[0];

  return (
    <section className="rounded-xl border border-border bg-card/70 p-4">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-sm font-semibold">Spoiler Tier</p>
          <p className="mt-1 text-sm text-muted-foreground">
            Retrieval is limited to passages at or below the selected tier.
          </p>
        </div>
        <div className="rounded-full border border-border px-3 py-1 text-xs font-medium uppercase tracking-wide text-foreground">
          {activeTier.label}
        </div>
      </div>

      <div className="mt-4">
        <input
          aria-label="Spoiler tier"
          className="w-full accent-primary"
          max={3}
          min={0}
          onChange={(event) => onChange(Number(event.target.value))}
          step={1}
          type="range"
          value={value}
        />
        <div className="mt-2 grid grid-cols-4 gap-2 text-xs text-muted-foreground">
          {SPOILER_TIERS.map((tier) => (
            <button
              key={tier.value}
              className={`rounded-md border px-2 py-2 text-left transition-colors ${
                value === tier.value
                  ? "border-primary bg-primary/10 text-foreground"
                  : "border-border hover:border-primary/40 hover:bg-muted"
              }`}
              onClick={() => onChange(tier.value)}
              type="button"
            >
              <span className="block font-medium">{tier.value}</span>
              <span className="mt-1 block">{tier.label}</span>
            </button>
          ))}
        </div>
        <p className="mt-3 text-sm text-muted-foreground">{activeTier.description}</p>
      </div>
    </section>
  );
}
