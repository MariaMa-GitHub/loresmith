"use client";

import { useState, useRef, useEffect } from "react";
import { streamChat, type ChatMessage, type Citation } from "@/lib/api";
import { MessageBubble } from "./MessageBubble";
import { SpoilerSlider } from "./SpoilerSlider";

interface ChatViewProps {
  gameSlug: string;
  gameDisplayName: string;
  sessionId?: string | null;
  initialMessages?: ChatMessage[];
  onSessionCreated?: (sessionId: string) => void;
  onMessagesChange?: (messages: ChatMessage[]) => void;
}

export function ChatView({
  gameSlug,
  gameDisplayName,
  sessionId = null,
  initialMessages = [],
  onSessionCreated,
  onMessagesChange,
}: ChatViewProps) {
  const [messages, setMessages] = useState<ChatMessage[]>(initialMessages);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [spoilerTier, setSpoilerTier] = useState(3);
  const bottomRef = useRef<HTMLDivElement>(null);
  const activeRequestIdRef = useRef(0);
  const activeAbortControllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Replace the visible thread when the parent provides a new message list,
  // such as selecting a saved session or explicitly starting a new chat.
  useEffect(() => {
    activeRequestIdRef.current += 1;
    activeAbortControllerRef.current?.abort();
    activeAbortControllerRef.current = null;
    setMessages(initialMessages);
    setInput("");
    setStreaming(false);
  }, [initialMessages]);

  useEffect(() => {
    return () => {
      activeRequestIdRef.current += 1;
      activeAbortControllerRef.current?.abort();
      activeAbortControllerRef.current = null;
    };
  }, []);

  useEffect(() => {
    onMessagesChange?.(messages);
  }, [messages, onMessagesChange]);

  function isAbortError(error: unknown) {
    return error instanceof Error && error.name === "AbortError";
  }

  function markStreamInterrupted() {
    setMessages((prev) => {
      const updated = [...prev];
      const last = updated[updated.length - 1];
      if (!last || last.role !== "assistant") {
        return prev;
      }
      const interrupted = last.content.trim()
        ? `${last.content}\n\nResponse interrupted. Please try again.`
        : "Something went wrong. Please try again.";
      updated[updated.length - 1] = {
        role: "assistant",
        content: interrupted,
      };
      return updated;
    });
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const question = input.trim();
    if (!question || streaming) return;
    const history = messages;
    let streamFailed = false;
    const requestId = activeRequestIdRef.current + 1;
    const abortController = new AbortController();

    activeRequestIdRef.current = requestId;
    activeAbortControllerRef.current?.abort();
    activeAbortControllerRef.current = abortController;

    setInput("");
    setStreaming(true);
    setMessages((prev) => [
      ...prev,
      { role: "user", content: question },
      { role: "assistant", content: "" },
    ]);

    try {
      for await (const event of streamChat({
        game: gameSlug,
        question,
        spoilerTier,
        sessionId,
        history,
        signal: abortController.signal,
      })) {
        if (requestId !== activeRequestIdRef.current) {
          abortController.abort();
          break;
        }

        if (event.type === "token" && typeof event.content === "string") {
          if (streamFailed) {
            continue;
          }
          setMessages((prev) => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (!last || last.role !== "assistant") {
              return prev;
            }
            updated[updated.length - 1] = {
              ...last,
              content: last.content + event.content,
            };
            return updated;
          });
        }

        if (event.type === "citations" && Array.isArray(event.content)) {
          if (streamFailed) {
            continue;
          }
          const citations = event.content as Citation[];
          setMessages((prev) => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (!last || last.role !== "assistant") {
              return prev;
            }
            updated[updated.length - 1] = {
              ...last,
              citations,
            };
            return updated;
          });
        }

        if (event.type === "answer" && typeof event.content === "string") {
          if (streamFailed) {
            continue;
          }
          const answer = event.content;
          setMessages((prev) => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (!last || last.role !== "assistant") {
              return prev;
            }
            updated[updated.length - 1] = {
              ...last,
              content: answer,
            };
            return updated;
          });
        }

        if (event.type === "error") {
          streamFailed = true;
          markStreamInterrupted();
          continue;
        }

        if (event.type === "session_id" && typeof event.content === "string") {
          onSessionCreated?.(event.content);
        }

        if (event.type === "done" && event.status === "error" && !streamFailed) {
          streamFailed = true;
          markStreamInterrupted();
        }
      }
    } catch (error) {
      if (requestId !== activeRequestIdRef.current || isAbortError(error)) {
        return;
      }
      markStreamInterrupted();
    } finally {
      if (requestId === activeRequestIdRef.current) {
        activeAbortControllerRef.current = null;
        setStreaming(false);
      }
    }
  }

  return (
    <div className="flex flex-col h-[calc(100vh-4rem)] max-w-2xl mx-auto p-4">
      <div className="mb-4">
        <SpoilerSlider onChange={setSpoilerTier} value={spoilerTier} />
      </div>
      <div className="flex-1 overflow-y-auto">
        {messages.length === 0 && (
          <p className="text-center text-muted-foreground mt-16 text-sm">
            Ask anything about {gameDisplayName} lore.
          </p>
        )}
        {messages.map((msg, i) => (
          <MessageBubble
            key={i}
            role={msg.role}
            content={msg.content}
            citations={msg.citations}
          />
        ))}
        <div ref={bottomRef} />
      </div>

      <form onSubmit={handleSubmit} className="mt-4 flex gap-2">
        <input
          className="flex-1 rounded-md border border-input bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about lore..."
          disabled={streaming}
          aria-label="Chat input"
        />
        <button
          type="submit"
          disabled={streaming || !input.trim()}
          className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground disabled:opacity-50"
        >
          {streaming ? "..." : "Send"}
        </button>
      </form>
    </div>
  );
}
