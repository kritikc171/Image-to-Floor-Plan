/** Status indicator with accessible live region. */

export function StatusBadge(props: { status: string; children: React.ReactNode }) {
  return (
    <span
      className={`status-badge status-badge--${props.status}`}
      role="status"
      aria-live="polite"
    >
      <span className="status-badge__pulse" aria-hidden="true" />
      {props.children}
    </span>
  );
}
