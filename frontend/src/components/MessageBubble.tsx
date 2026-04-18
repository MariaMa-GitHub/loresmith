import { Fragment, type ReactNode } from "react";

import type { Citation } from "@/lib/api";

interface MessageBubbleProps {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
}

const CITATION_GROUP_RE = /\[(\d+(?:\s*,\s*\d+)*)\]/g;

function getCitationHost(sourceUrl: string): string {
  try {
    return new URL(sourceUrl).hostname.replace(/^www\./, "");
  } catch {
    return sourceUrl;
  }
}

function getCitationTitle(citation: Citation): string {
  if (citation.title?.trim()) {
    return citation.title.trim();
  }
  return citation.source_url;
}

function renderCitationGroup(
  matchText: string,
  citationsByIndex: Map<number, Citation>,
  key: string,
) {
  const indices = matchText
    .slice(1, -1)
    .split(",")
    .map((value) => Number(value.trim()))
    .filter((value) => Number.isInteger(value));

  if (indices.length === 0) {
    return <Fragment key={key}>{matchText}</Fragment>;
  }

  return (
    <Fragment key={key}>
      [
      {indices.map((citationIndex, index) => {
        const citation = citationsByIndex.get(citationIndex);
        const separator = index > 0 ? ", " : "";

        return (
          <Fragment key={`${key}-${citationIndex}`}>
            {separator}
            {citation ? (
              <a
                href={citation.source_url}
                target="_blank"
                rel="noreferrer"
                className="underline underline-offset-2 hover:text-primary"
                title={citation.source_url}
              >
                {citationIndex}
              </a>
            ) : (
              citationIndex
            )}
          </Fragment>
        );
      })}
      ]
    </Fragment>
  );
}

function renderContent(content: string, citations: Citation[] = []) {
  const orderedCitations = [...citations].sort((left, right) => left.index - right.index);
  const citationsByIndex = new Map(
    orderedCitations.map((citation) => [citation.index, citation]),
  );
  const nodes: ReactNode[] = [];
  let cursor = 0;

  for (const match of content.matchAll(CITATION_GROUP_RE)) {
    const start = match.index ?? 0;
    const fullMatch = match[0];

    if (start > cursor) {
      nodes.push(
        <Fragment key={`text-${cursor}`}>
          {content.slice(cursor, start)}
        </Fragment>,
      );
    }

    nodes.push(
      renderCitationGroup(fullMatch, citationsByIndex, `citation-${start}`),
    );
    cursor = start + fullMatch.length;
  }

  if (cursor < content.length) {
    nodes.push(
      <Fragment key={`text-${cursor}`}>
        {content.slice(cursor)}
      </Fragment>,
    );
  }

  return nodes;
}

export function MessageBubble({ role, content, citations = [] }: MessageBubbleProps) {
  const isUser = role === "user";
  const orderedCitations = [...citations].sort((left, right) => left.index - right.index);
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-3`}>
      <div
        className={`max-w-[80%] rounded-lg px-4 py-2 text-sm ${
          isUser
            ? "bg-primary text-primary-foreground"
            : "bg-muted text-foreground"
        }`}
      >
        <p className="whitespace-pre-wrap">{renderContent(content, orderedCitations)}</p>
        {!isUser && orderedCitations.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2 border-t border-border/60 pt-3">
            {orderedCitations.map((citation) => (
              <a
                key={citation.index}
                href={citation.source_url}
                target="_blank"
                rel="noreferrer"
                className="min-w-0 rounded-lg border border-border px-3 py-2 text-left hover:border-primary hover:bg-background/60"
                title={`${getCitationTitle(citation)}\n${citation.source_url}`}
              >
                <div className="truncate text-xs font-medium text-foreground">
                  [{citation.index}] {getCitationTitle(citation)}
                </div>
                <div className="mt-1 truncate text-[10px] text-muted-foreground">
                  {getCitationHost(citation.source_url)}
                </div>
              </a>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
