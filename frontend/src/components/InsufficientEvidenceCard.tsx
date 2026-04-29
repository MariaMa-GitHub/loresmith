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
          <ul className="mt-1 list-disc pl-5 space-y-1 text-xs">
            {rewriteSuggestions.map((suggestion, i) => (
              <li key={i}>{suggestion}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
