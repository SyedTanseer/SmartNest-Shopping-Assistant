import { inr } from "../format";
import type { LiveStatus, PlanItem, RoomView } from "../types";
import { Badge, Empty } from "../ui";
import { ItemCard, type ItemActions } from "./ItemCard";

interface Props extends ItemActions {
  rooms: RoomView[];
  shortlist: string[];
  cartIds: string[];
  busy: boolean;
  notes: string[];
  assumptions: string[];
  live?: LiveStatus;
  budgetNeeded?: number | null;
  onRaiseBudget?: (amount: number) => void;
  onSearchMany?: (categories: string[]) => void;
}

export function PlanView({ rooms, shortlist, cartIds, busy, notes, assumptions, live, budgetNeeded, onRaiseBudget, onSearchMany, ...actions }: Props) {
  if (rooms.length === 0) {
    return <Empty title="No plan yet">Tell the assistant what to furnish — for example “2BHK, ₹2 lakh, minimalist” or “my son's bedroom under 50000”.</Empty>;
  }
  // Categories whose items cannot be answered from saved data: nothing saved, or no saved listing fits the theme.
  const needing = Array.from(new Set(rooms.flatMap((r) => r.items as PlanItem[]).filter((i) => i.no_data || i.theme_unmatched).map((i) => i.category)));
  return (
    <div className="space-y-6">
      {live?.enabled && onSearchMany && needing.length >= 2 && (
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-brand/30 bg-brand-soft px-4 py-3 text-sm" data-testid="search-many">
          <span><strong>{needing.length} items</strong> need a live search (nothing saved, or nothing matching your theme).</span>
          <button type="button" disabled={busy} onClick={() => onSearchMany(needing)}
            className="rounded-lg bg-fill px-3 py-1.5 font-medium text-white hover:bg-fill-hover disabled:opacity-60">
            Search live for all {needing.length} ({needing.length} requests)
          </button>
        </div>
      )}
      {rooms.map((room) => (
        <section key={room.room_id} aria-labelledby={`room-${room.room_id}`}>
          <header className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
            <h3 id={`room-${room.room_id}`} className="text-lg font-semibold">{room.label}</h3>
            <div className="flex items-center gap-2 text-sm text-muted">
              <span>{room.filled}/{room.count} filled</span>
              <Badge tone="brand">{inr(room.spent)}</Badge>
            </div>
          </header>
          <div className="grid gap-3 xl:grid-cols-2">
            {room.items.map((item) => (
              <ItemCard key={item.slot_id} item={item} shortlist={shortlist} cartIds={cartIds} busy={busy} live={live} raiseTo={budgetNeeded} onRaiseBudget={onRaiseBudget} {...actions} />
            ))}
          </div>
        </section>
      ))}
      {(assumptions.length > 0 || notes.length > 0) && (
        <details className="rounded-xl border border-line bg-surface p-4 text-sm">
          <summary className="cursor-pointer font-medium">Assumptions and data notes</summary>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-muted">
            {[...assumptions, ...notes].map((n, i) => <li key={i}>{n}</li>)}
          </ul>
        </details>
      )}
    </div>
  );
}
