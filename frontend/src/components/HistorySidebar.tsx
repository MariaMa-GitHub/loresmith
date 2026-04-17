"use client";

interface Session {
  id: string;
  preview: string;
  createdAt: string;
}

interface HistorySidebarProps {
  sessions?: Session[];
  currentSessionId?: string;
  onSelect?: (id: string) => void;
}

export function HistorySidebar({
  sessions = [],
  currentSessionId,
  onSelect,
}: HistorySidebarProps) {
  return (
    <aside className="w-64 border-r border-border h-full flex flex-col">
      <div className="p-4 border-b border-border">
        <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
          History
        </h2>
      </div>
      <nav className="flex-1 overflow-y-auto p-2">
        {sessions.length === 0 && (
          <p className="text-xs text-muted-foreground px-2 py-4 text-center">
            No previous conversations
          </p>
        )}
        {sessions.map((session) => (
          <button
            key={session.id}
            onClick={() => onSelect?.(session.id)}
            aria-current={session.id === currentSessionId ? "page" : undefined}
            className={`w-full text-left rounded-md px-3 py-2 text-sm mb-1 truncate ${
              session.id === currentSessionId
                ? "bg-primary/10 text-foreground"
                : "text-muted-foreground hover:bg-muted hover:text-foreground"
            }`}
          >
            {session.preview || "Untitled session"}
          </button>
        ))}
      </nav>
    </aside>
  );
}
