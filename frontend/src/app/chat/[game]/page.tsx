import Link from "next/link";
import { notFound } from "next/navigation";
import { ChatLayout } from "@/components/ChatLayout";
import { fetchGames } from "@/lib/api";

interface Props {
  params: Promise<{ game: string }>;
}

export default async function ChatPage({ params }: Props) {
  const { game } = await params;
  const games = await fetchGames();
  const selectedGame = games.find((candidate) => candidate.slug === game);
  if (!selectedGame) {
    notFound();
  }

  return (
    <div className="flex h-screen flex-col">
      <header className="border-b border-border px-4 py-3 flex items-center gap-3 flex-shrink-0">
        <Link href="/" aria-label="Back to games" className="text-sm text-muted-foreground hover:text-foreground">
          ← Games
        </Link>
        <span className="text-sm font-medium">{selectedGame.display_name}</span>
      </header>
      <ChatLayout
        key={game}
        gameSlug={game}
        gameDisplayName={selectedGame.display_name}
      />
    </div>
  );
}
