import { fetchGames, type Game } from "@/lib/api";
import { GamePicker } from "@/components/GamePicker";

export default async function Home() {
  let games: Game[] = [];
  try {
    games = await fetchGames();
  } catch {
    // Backend not running locally — show placeholder
    games = [{ slug: "hades", display_name: "Hades" }];
  }

  return (
    <main className="min-h-screen p-8 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold tracking-tight">Loresmith</h1>
      <p className="mt-2 text-muted-foreground">
        Answers about video-game lore, grounded in source material with inline citations.
      </p>
      <GamePicker games={games} />
    </main>
  );
}
