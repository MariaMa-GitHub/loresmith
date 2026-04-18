"use client";

import { useCallback, useRef, useState } from "react";
import { ChatView } from "./ChatView";
import { HistorySidebar } from "./HistorySidebar";
import { fetchSessionMessages, type ChatMessage } from "@/lib/api";

interface ChatLayoutProps {
  gameSlug: string;
  gameDisplayName: string;
}

export function ChatLayout({ gameSlug, gameDisplayName }: ChatLayoutProps) {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [initialMessages, setInitialMessages] = useState<ChatMessage[]>([]);
  const [sidebarRefresh, setSidebarRefresh] = useState(0);
  const [loadError, setLoadError] = useState<string | null>(null);
  const sessionCacheRef = useRef<Map<string, ChatMessage[]>>(new Map());
  const selectionNonceRef = useRef(0);

  const handleSelect = useCallback(
    async (id: string) => {
      if (id === sessionId) return;
      selectionNonceRef.current += 1;
      const selectionNonce = selectionNonceRef.current;
      setLoadError(null);

      const cached = sessionCacheRef.current.get(id);
      if (cached) {
        setSessionId(id);
        setInitialMessages(cached);
        return;
      }

      try {
        const data = await fetchSessionMessages(id);
        if (selectionNonce !== selectionNonceRef.current) {
          return;
        }
        if (data.game_slug !== gameSlug) {
          setLoadError("That conversation belongs to a different game.");
          return;
        }
        sessionCacheRef.current.set(id, data.messages);
        setSessionId(id);
        setInitialMessages(data.messages);
      } catch (err) {
        if (selectionNonce !== selectionNonceRef.current) {
          return;
        }
        setLoadError(
          err instanceof Error ? err.message : "Could not load session",
        );
      }
    },
    [gameSlug, sessionId],
  );

  const handleNew = useCallback(() => {
    selectionNonceRef.current += 1;
    setSessionId(null);
    setInitialMessages([]);
    setLoadError(null);
  }, []);

  const handleSessionCreated = useCallback(
    (id: string) => {
      if (id !== sessionId) {
        setSessionId(id);
      }
      setSidebarRefresh((value) => value + 1);
    },
    [sessionId],
  );

  const handleMessagesChange = useCallback(
    (messages: ChatMessage[]) => {
      if (!sessionId) {
        return;
      }
      sessionCacheRef.current.set(sessionId, messages);
    },
    [sessionId],
  );

  return (
    <div className="flex flex-1 overflow-hidden">
      <HistorySidebar
        key={`${gameSlug}:${sidebarRefresh}`}
        game={gameSlug}
        currentSessionId={sessionId}
        onSelect={handleSelect}
        onNew={handleNew}
      />
      <main className="flex-1 overflow-hidden">
        {loadError && (
          <p
            role="alert"
            className="text-xs text-destructive px-4 py-2 border-b border-border"
          >
            {loadError}
          </p>
        )}
        <ChatView
          gameSlug={gameSlug}
          gameDisplayName={gameDisplayName}
          sessionId={sessionId}
          initialMessages={initialMessages}
          onSessionCreated={handleSessionCreated}
          onMessagesChange={handleMessagesChange}
        />
      </main>
    </div>
  );
}
