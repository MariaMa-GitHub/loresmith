interface Props {
  message: string;
  rewriteSuggestions: string[];
  unsupportedClaims: string[];
}

export function InsufficientEvidenceCard({
  message,
  rewriteSuggestions,
  unsupportedClaims,
}: Props) {
  return (
    <div
      role="status"
      aria-live="polite"
      className="rounded-md border border-yellow-500/40 bg-yellow-500/10 p-4 text-sm text-foreground"
    >
      <p className="font-medium">Insufficient evidence</p>
      <p className="mt-1 text-muted-foreground">{message}</p>

      {unsupportedClaims.length > 0 && (
        <div className="mt-3">
          <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
            Unsupported claims
          </p>
          <ul className="mt-1 list-disc pl-5 text-xs">
            {unsupportedClaims.map((claim, i) => (
              <li key={i}>{claim}</li>
            ))}
          </ul>
        </div>
      )}

      {rewriteSuggestions.length > 0 && (
        <div className="mt-3">
          <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
            Try asking
          </p>
          <ul className="mt-1 space-y-1">
            {rewriteSuggestions.map((suggestion, i) => (
              <li key={i}>
                <button
                  type="button"
                  className="w-full rounded border border-border px-2 py-1 text-left text-xs hover:border-primary"
                >
                  {suggestion}
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
