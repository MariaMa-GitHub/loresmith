import Link from "next/link";
import { ChatView } from "@/components/ChatView";
import { HistorySidebar } from "@/components/HistorySidebar";

interface Props {
  params: Promise<{ game: string }>;
}

export default async function ChatPage({ params }: Props) {
  const { game } = await params;
  return (
    <div className="flex h-screen flex-col">
      <header className="border-b border-border px-4 py-3 flex items-center gap-3 flex-shrink-0">
        <Link href="/" aria-label="Back to games" className="text-sm text-muted-foreground hover:text-foreground">
          ← Games
        </Link>
        <span className="text-sm font-medium capitalize">{game}</span>
      </header>
      <div className="flex flex-1 overflow-hidden">
        <HistorySidebar />
        <main className="flex-1 overflow-hidden">
          <ChatView game={game} />
        </main>
      </div>
    </div>
  );
}
