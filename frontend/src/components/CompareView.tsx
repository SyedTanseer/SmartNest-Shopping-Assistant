import { useEffect, useState } from "react";
import { api } from "../api";
import { COMPONENT_LABEL, inr, percent } from "../format";
import type { CompareProduct, PlanItem } from "../types";
import { Badge, Button, Card, Empty } from "../ui";
import { PurchaseLink } from "./ItemCard";

interface Props {
  sessionId: string;
  items: PlanItem[];
  shortlist: string[];
  onToggle: (productId: string) => void;
  cartIds: string[];
  onAddToCart: (productId: string) => void;
}

/** Name and slot of every product the plan knows about, so the shortlist can show names. */
function known(items: PlanItem[]) {
  const map = new Map<string, { name: string; price: number; label: string }>();
  for (const it of items) {
    for (const p of [it.product, ...it.alternatives]) {
      if (p) map.set(p.product_id, { name: p.name, price: p.price, label: it.label });
    }
  }
  return map;
}

export function CompareView({ sessionId, items, shortlist, onToggle, cartIds, onAddToCart }: Props) {
  const [picked, setPicked] = useState<string[]>([]);
  const [rows, setRows] = useState<CompareProduct[]>([]);
  const [error, setError] = useState<string | null>(null);
  const names = known(items);

  // Keep the selection to products still on the shortlist.
  useEffect(() => setPicked((p) => p.filter((id) => shortlist.includes(id))), [shortlist]);

  useEffect(() => {
    if (picked.length < 2) { setRows([]); setError(null); return; }
    let live = true;
    api.compare(sessionId, picked)
      .then((r) => { if (live) { setRows(r.products); setError(null); } })
      .catch((e: Error) => { if (live) setError(e.message); });
    return () => { live = false; };
  }, [picked, sessionId]);

  if (shortlist.length === 0) {
    return <Empty title="Your shortlist is empty">Press “☆ Shortlist” on products in the plan, then tick two to four to compare them side by side.</Empty>;
  }
  const toggle = (id: string) =>
    setPicked((p) => (p.includes(id) ? p.filter((x) => x !== id) : p.length >= 4 ? p : [...p, id]));
  const keys = Array.from(new Set(rows.flatMap((r) => Object.keys(r.contributions ?? {}))));

  return (
    <div className="space-y-4">
      <Card className="p-4">
        <h3 className="font-semibold">Shortlist ({shortlist.length})</h3>
        <ul className="mt-2 divide-y divide-line">
          {shortlist.map((id) => {
            const info = names.get(id);
            return (
              <li key={id} className="flex items-center justify-between gap-3 py-2 text-sm">
                <label className="flex min-w-0 items-center gap-2">
                  <input type="checkbox" checked={picked.includes(id)} onChange={() => toggle(id)}
                    disabled={!picked.includes(id) && picked.length >= 4} aria-label={`Compare ${info?.name ?? id}`} />
                  <span className="min-w-0"><span className="block truncate font-medium">{info?.name ?? id}</span>
                    <span className="text-xs text-muted">{info?.label}</span></span>
                </label>
                <div className="flex items-center gap-2"><span className="tabular-nums">{inr(info?.price)}</span>
                  <Button disabled={cartIds.includes(id)} onClick={() => onAddToCart(id)} aria-label={`Add ${info?.name ?? id} to cart`}>{cartIds.includes(id) ? "✓ In cart" : "Add to cart"}</Button>
                  <Button onClick={() => onToggle(id)} aria-label={`Remove ${info?.name ?? id} from shortlist`}>Remove</Button></div>
              </li>
            );
          })}
        </ul>
      </Card>

      {picked.length < 2 && <p className="text-sm text-muted">Tick at least two products to compare.</p>}
      {error && <p role="alert" className="text-sm text-bad">{error}</p>}

      {rows.length >= 2 && (
        <Card className="overflow-x-auto">
          <table className="w-full min-w-[34rem] text-left text-sm">
            <caption className="sr-only">Side-by-side comparison</caption>
            <thead>
              <tr className="border-b border-line align-bottom">
                <th scope="col" className="w-32 px-4 py-3 text-xs uppercase tracking-wide text-muted">Product</th>
                {rows.map((r) => <th scope="col" key={r.product_id} className="px-4 py-3 font-medium">{r.name}</th>)}
              </tr>
            </thead>
            <tbody>
              <Row label="For">{rows.map((r) => <td key={r.product_id} className="px-4 py-2">{r.item}{r.is_pick && <> <Badge tone="brand">SmartNest pick</Badge></>}</td>)}</Row>
              <Row label="Price">{rows.map((r) => <td key={r.product_id} className="px-4 py-2 font-semibold tabular-nums">{inr(r.price)}</td>)}</Row>
              <Row label="Seller">{rows.map((r) => <td key={r.product_id} className="px-4 py-2">{r.retailer ?? "—"}</td>)}</Row>
              <Row label="Rating">{rows.map((r) => <td key={r.product_id} className="px-4 py-2">{r.rating !== null ? `★ ${r.rating}` : "not stated"}</td>)}</Row>
              <Row label="Utility">{rows.map((r) => <td key={r.product_id} className="px-4 py-2 tabular-nums">{r.utility?.toFixed(1) ?? "—"}/100</td>)}</Row>
              <Row label="Score basis">{rows.map((r) => <td key={r.product_id} className="px-4 py-2">{percent(r.completeness)} of criteria</td>)}</Row>
              {keys.map((k) => (
                <Row key={k} label={`${COMPONENT_LABEL[k] ?? k} points`}>
                  {rows.map((r) => <td key={r.product_id} className="px-4 py-2 tabular-nums text-muted">{r.contributions?.[k]?.toFixed(1) ?? "—"}</td>)}
                </Row>
              ))}
              <Row label="Stock">{rows.map((r) => <td key={r.product_id} className="px-4 py-2">
                {r.state === "confirmed" ? <Badge tone="ok">Confirmed</Badge> : <span className="text-muted">Not stated</span>}</td>)}</Row>
              <Row label="Link">{rows.map((r) => <td key={r.product_id} className="px-4 py-2"><PurchaseLink product={r} /></td>)}</Row>
            </tbody>
          </table>
        </Card>
      )}
    </div>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return <tr className="border-t border-line"><th scope="row" className="px-4 py-2 text-xs font-medium uppercase tracking-wide text-muted">{label}</th>{children}</tr>;
}
