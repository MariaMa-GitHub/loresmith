const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface Game {
  slug: string;
  display_name: string;
}

export async function fetchGames(): Promise<Game[]> {
  const res = await fetch(`${API_BASE}/games`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch games");
  const data = await res.json();
  return data.games as Game[];
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
}

interface ChatHistoryMessage {
  role: ChatMessage["role"];
  content: string;
}

export interface Citation {
  index: number;
  passage_id?: number;
  source_url: string;
  title?: string;
  content?: string;
}

export interface ChatEvent {
  type: string;
  content?: unknown;
  status?: string;
}

export interface SessionSummary {
  id: string;
  game_slug: string;
  preview: string;
  updated_at: string | null;
}

export async function fetchSessions(
  game: string,
  limit: number = 20,
): Promise<SessionSummary[]> {
  const res = await fetch(
    `${API_BASE}/sessions/${encodeURIComponent(game)}?limit=${limit}`,
    { cache: "no-store", credentials: "include" },
  );
  if (!res.ok) throw new Error(`Failed to fetch sessions: ${res.status}`);
  const data = await res.json();
  return data.sessions as SessionSummary[];
}

export async function fetchSessionMessages(
  sessionId: string,
): Promise<{ messages: ChatMessage[]; game_slug: string }> {
  const res = await fetch(
    `${API_BASE}/sessions/${encodeURIComponent(sessionId)}/messages`,
    { cache: "no-store", credentials: "include" },
  );
  if (!res.ok) throw new Error(`Failed to fetch session: ${res.status}`);
  const data = await res.json();
  return {
    game_slug: data.game_slug,
    messages: (data.messages as ChatMessage[]).map((msg) => ({
      role: msg.role,
      content: msg.content,
      citations: msg.citations,
    })),
  };
}

interface StreamChatOptions {
  game: string;
  question: string;
  spoilerTier?: number;
  sessionId?: string | null;
  history?: ChatMessage[];
  signal?: AbortSignal;
}

export async function* streamChat({
  game,
  question,
  spoilerTier = 0,
  sessionId,
  history = [],
  signal,
}: StreamChatOptions): AsyncGenerator<ChatEvent> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    credentials: "include",
    signal,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      game,
      question,
      spoiler_tier: spoilerTier,
      session_id: sessionId ?? undefined,
      history: history.map<ChatHistoryMessage>(({ role, content }) => ({
        role,
        content,
      })),
    }),
  });

  if (!res.ok) throw new Error(`Chat request failed: ${res.status}`);
  if (!res.body) throw new Error("No response body");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const payload = JSON.parse(line.slice(6)) as ChatEvent;
      yield payload;
      if (payload.type === "done") return;
    }
  }
}
