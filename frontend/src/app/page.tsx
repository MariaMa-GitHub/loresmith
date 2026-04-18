import { fetchGames, type Game } from "@/lib/api";
import { GamePicker } from "@/components/GamePicker";

export const dynamic = "force-dynamic";

export default async function Home() {
  let games: Game[] = [];
  let loadError = false;
  try {
    games = await fetchGames();
  } catch {
    loadError = true;
  }

  return (
    <main className="min-h-screen p-8 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold tracking-tight">Loresmith</h1>
      <p className="mt-2 text-muted-foreground">
        Answers about video-game lore, grounded in source material with inline citations.
      </p>
      {loadError ? (
        <p className="mt-8 rounded-lg border border-border bg-muted/40 p-4 text-sm text-muted-foreground">
          The game catalog is unavailable right now. Check that the backend is running and try
          again.
        </p>
      ) : (
        <GamePicker games={games} />
      )}
    </main>
  );
}
