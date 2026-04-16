import Link from "next/link";
import { ChatView } from "@/components/ChatView";

interface Props {
  params: Promise<{ game: string }>;
}

export default async function ChatPage({ params }: Props) {
  const { game } = await params;
  return (
    <div>
      <header className="border-b border-border px-4 py-3 flex items-center gap-3">
        <Link href="/" className="text-sm text-muted-foreground hover:text-foreground">
          ← Games
        </Link>
        <span className="text-sm font-medium capitalize">{game}</span>
      </header>
      <ChatView game={game} />
    </div>
  );
}
