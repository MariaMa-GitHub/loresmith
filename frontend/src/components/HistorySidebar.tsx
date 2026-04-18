"use client";

import { useEffect, useState } from "react";
import { fetchSessions, type SessionSummary } from "@/lib/api";

interface HistorySidebarProps {
  game: string;
  currentSessionId?: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
}

function formatRelative(iso: string | null): string {
  if (!iso) return "";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "";
  const diffMs = Date.now() - then;
  const minute = 60_000;
  const hour = 60 * minute;
  const day = 24 * hour;
  if (diffMs < minute) return "just now";
  if (diffMs < hour) return `${Math.floor(diffMs / minute)}m ago`;
  if (diffMs < day) return `${Math.floor(diffMs / hour)}h ago`;
  return `${Math.floor(diffMs / day)}d ago`;
}

export function HistorySidebar({
  game,
  currentSessionId,
  onSelect,
  onNew,
}: HistorySidebarProps) {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchSessions(game)
      .then((data) => {
        if (!cancelled) {
          setSessions(data);
          setError(null);
          setLoading(false);
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setSessions([]);
          setError(err instanceof Error ? err.message : "Failed to load");
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [game]);

  return (
    <aside className="w-64 border-r border-border h-full flex flex-col">
      <div className="p-4 border-b border-border flex items-center justify-between">
        <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
          History
        </h2>
        <button
          type="button"
          onClick={onNew}
          className="text-xs rounded-md px-2 py-1 text-muted-foreground hover:bg-muted hover:text-foreground"
          aria-label="Start a new chat"
        >
          + New
        </button>
      </div>
      <nav className="flex-1 overflow-y-auto p-2">
        {loading && (
          <p className="text-xs text-muted-foreground px-2 py-4 text-center">
            Loading…
          </p>
        )}
        {!loading && error && (
          <p className="text-xs text-destructive px-2 py-4 text-center">
            {error}
          </p>
        )}
        {!loading && !error && sessions.length === 0 && (
          <p className="text-xs text-muted-foreground px-2 py-4 text-center">
            No previous conversations
          </p>
        )}
        {!loading &&
          !error &&
          sessions.map((session) => {
            const isCurrent = session.id === currentSessionId;
            return (
              <button
                key={session.id}
                onClick={() => onSelect(session.id)}
                aria-current={isCurrent ? "page" : undefined}
                className={`w-full text-left rounded-md px-3 py-2 text-sm mb-1 ${
                  isCurrent
                    ? "bg-primary/10 text-foreground"
                    : "text-muted-foreground hover:bg-muted hover:text-foreground"
                }`}
              >
                <div className="truncate">
                  {session.preview || "Untitled session"}
                </div>
                <div className="text-[10px] text-muted-foreground mt-0.5">
                  {formatRelative(session.updated_at)}
                </div>
              </button>
            );
          })}
      </nav>
    </aside>
  );
}
