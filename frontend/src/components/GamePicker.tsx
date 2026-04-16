"use client";

import { useRouter } from "next/navigation";
import { type Game } from "@/lib/api";

interface GamePickerProps {
  games: Game[];
}

export function GamePicker({ games }: GamePickerProps) {
  const router = useRouter();

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 mt-8">
      {games.map((game) => (
        <button
          key={game.slug}
          onClick={() => router.push(`/chat/${game.slug}`)}
          className="rounded-lg border border-border p-6 text-left hover:border-primary hover:bg-muted transition-colors"
        >
          <h2 className="text-lg font-semibold">{game.display_name}</h2>
          <p className="mt-1 text-sm text-muted-foreground">Ask about lore →</p>
        </button>
      ))}
    </div>
  );
}
