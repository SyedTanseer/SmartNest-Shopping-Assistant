/** ₹ with Indian digit grouping: 1234567 -> ₹12,34,567. */
export function inr(amount: number | null | undefined): string {
  if (amount === null || amount === undefined || Number.isNaN(amount)) return "—";
  const n = Math.round(amount);
  const sign = n < 0 ? "-" : "";
  let s = String(Math.abs(n));
  if (s.length > 3) {
    let head = s.slice(0, -3);
    const tail = s.slice(-3);
    const parts: string[] = [];
    while (head.length > 2) {
      parts.unshift(head.slice(-2));
      head = head.slice(0, -2);
    }
    if (head) parts.unshift(head);
    s = [...parts, tail].join(",");
  }
  return `${sign}₹${s}`;
}

/** Compact form for chart axes: 200000 -> ₹2L, 45000 -> ₹45k. */
export function inrShort(amount: number): string {
  if (amount >= 1e7) return `₹${+(amount / 1e7).toFixed(1)}Cr`;
  if (amount >= 1e5) return `₹${+(amount / 1e5).toFixed(1)}L`;
  if (amount >= 1e3) return `₹${+(amount / 1e3).toFixed(0)}k`;
  return `₹${Math.round(amount)}`;
}

export const percent = (x: number | null | undefined): string =>
  x === null || x === undefined ? "—" : `${Math.round(x * 100)}%`;

export function timeAgo(iso: string | null | undefined, now: Date = new Date()): string {
  if (!iso) return "unknown";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "unknown";
  const hours = (now.getTime() - then) / 3.6e6;
  if (hours < 1) return "under an hour ago";
  if (hours < 48) return `${Math.round(hours)} h ago`;
  return `${Math.round(hours / 24)} days ago`;
}

export const STATUS_LABEL = {
  confirmed: "Confirmed in stock",
  unverified: "Stock unverified",
  unfilled: "Not filled",
} as const;

export const COMPONENT_LABEL: Record<string, string> = {
  price: "Price", quality: "Rating", style: "Style", colour: "Colour", spatial: "Fit",
};
