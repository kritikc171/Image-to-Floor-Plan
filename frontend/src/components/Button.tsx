/** Primary action button. */
export function Button(props: {
  children: React.ReactNode;
  onClick?: () => void;
  variant?: "primary" | "secondary";
  disabled?: boolean;
  type?: "button" | "submit";
  className?: string;
}) {
  const base: React.CSSProperties = {
    appearance: "none",
    border: 0,
    borderRadius: "var(--radius)",
    padding: "10px 18px",
    fontSize: "0.95rem",
    fontWeight: 600,
    cursor: props.disabled ? "not-allowed" : "pointer",
    transition: "background 120ms ease",
    background:
      props.variant === "secondary" ? "var(--surface)" : "var(--accent)",
    color:
      props.variant === "secondary" ? "var(--text)" : "var(--accent-contrast)",
    boxShadow: "var(--shadow-sm)",
    opacity: props.disabled ? 0.55 : 1,
    ...({
      border: props.variant === "secondary" ? "1px solid var(--border-strong)" : undefined,
    } as React.CSSProperties),
  };
  return (
    <button
      type={props.type ?? "button"}
      style={base}
      disabled={props.disabled}
      onClick={props.onClick}
      className={props.className}
    >
      {props.children}
    </button>
  );
}
