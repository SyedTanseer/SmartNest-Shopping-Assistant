import { useState } from "react";
import { COMPONENT_LABEL, STATUS_LABEL, inr, percent, timeAgo } from "../format";
import type { Alternative, LiveStatus, PlanItem } from "../types";
import { Badge, Button, ScoreBar } from "../ui";

export interface ItemActions {
  onShortlist: (productId: string) => void;
  onChoose: (slotId: string, productId: string) => void;
  onClearChoice: (slotId: string) => void;
  onOutOfStock: (productId: string) => void;
  onRemove: (slotId: string) => void;
  onAddToCart: (productId: string) => void;
  onSearch: (category: string) => void;
}

interface Props extends ItemActions {
  item: PlanItem;
  shortlist: string[];
  cartIds: string[];
  busy: boolean;
  live?: LiveStatus;
  raiseTo?: number | null;                       // the budget that would fit every item at its cheapest
  onRaiseBudget?: (amount: number) => void;
}

function Thumb({ src, name }: { src: string | null; name: string }) {
  const [failed, setFailed] = useState(false);
  if (!src || failed) {
    return (
      <div aria-hidden className="flex h-24 w-24 shrink-0 items-center justify-center rounded-lg bg-subtle text-2xl text-muted">
        ⌂
      </div>
    );
  }
  return <img src={src} alt={name} loading="lazy" referrerPolicy="no-referrer" onError={() => setFailed(true)}
    className="h-24 w-24 shrink-0 rounded-lg bg-tile object-contain ring-1 ring-line" />;
}

export function PurchaseLink({ product }: { product: { buy_url: string | null; link_label: string; simulated: boolean } }) {
  if (product.simulated || !product.buy_url) {
    return <span className="text-xs text-muted">{product.link_label}</span>;
  }
  return (
    <a href={product.buy_url} target="_blank" rel="noopener noreferrer"
      className="text-sm font-medium text-brand underline-offset-2 hover:underline">
      {product.link_label} ↗
    </a>
  );
}

function AlternativeRow({ alt, slotId, busy, inCart, onChoose, onAddToCart }: { alt: Alternative; slotId: string; busy: boolean; inCart: boolean; onChoose: Props["onChoose"]; onAddToCart: Props["onAddToCart"] }) {
  return (
    <li className="flex items-center justify-between gap-3 py-2 text-sm">
      <div className="min-w-0">
        <div className="truncate font-medium" title={alt.name}>{alt.name}</div>
        <div className="text-xs text-muted">
          {alt.retailer ?? "Unknown seller"} · utility {alt.utility?.toFixed(1) ?? "—"}
          {alt.state === "confirmed" && " · in stock"}
        </div>
      </div>
      <div className="flex shrink-0 items-center gap-2">
        <span className="tabular-nums font-medium">{inr(alt.price)}</span>
        <Button disabled={busy} onClick={() => onChoose(slotId, alt.product_id)} aria-label={`Use ${alt.name}`}>Use this</Button>
        <Button disabled={busy || inCart} onClick={() => onAddToCart(alt.product_id)} aria-label={`Add ${alt.name} to cart`}>{inCart ? "✓ In cart" : "Add to cart"}</Button>
      </div>
    </li>
  );
}

export function ItemCard({ item, shortlist, cartIds, busy, live, raiseTo, onRaiseBudget, onShortlist, onChoose, onClearChoice, onOutOfStock, onRemove, onAddToCart, onSearch }: Props) {
  const [open, setOpen] = useState<"" | "alts" | "why">("");
  const p = item.product;

  if (!p) {
    return (
      <article className="rounded-xl border border-dashed border-bad/30 bg-bad-soft/40 p-4" data-testid={`item-${item.slot_id}`}>
        <div className="flex items-start justify-between gap-3">
          <div>
            <h4 className="font-medium">{item.category_label}</h4>
            <p className={`mt-1 text-sm ${item.no_data ? "text-ink" : "text-bad"}`}>{item.why}</p>
          </div>
          <Badge tone={item.no_data ? "warn" : "bad"}>{item.no_data ? "No listings yet" : STATUS_LABEL.unfilled}</Badge>
        </div>
        {item.no_data ? (
          <div className="mt-3 flex flex-wrap items-center gap-3">
            <Button variant="primary" disabled={busy || !live?.enabled || live.remaining === 0}
              onClick={() => onSearch(item.category)} aria-label={`Search live for ${item.category_label}`}>
              Search live (1 request)
            </Button>
            <span className="text-xs text-muted">
              {!live?.enabled ? "Live search is switched off (set SMARTNEST_ALLOW_LIVE_SEARCH=1)."
                : live.remaining === 0 ? `The SerpApi allowance is used up (${live.used} of ${live.limit}).`
                : live.remaining != null ? `${live.remaining} SerpApi request(s) left.` : "Uses one SerpApi request."}
              {item.search_query ? ` Searches for “${item.search_query}”.` : ""}
            </span>
          </div>
        ) : (
          <div className="mt-3 flex flex-wrap items-center gap-2">
            {raiseTo && onRaiseBudget && (
              <Button variant="primary" disabled={busy} onClick={() => onRaiseBudget(raiseTo)}>Raise budget to {inr(raiseTo)}</Button>
            )}
            <Button disabled={busy} onClick={() => onRemove(item.slot_id)} aria-label={`Remove ${item.category_label} from the plan`}>Remove item</Button>
            {item.cheapest_price != null && <span className="text-xs text-muted">Cheapest listing: {inr(item.cheapest_price)}</span>}
          </div>
        )}
      </article>
    );
  }

  const starred = shortlist.includes(p.product_id);
  const inCart = cartIds.includes(p.product_id);
  return (
    <article className="rounded-xl border border-line bg-card p-4 shadow-sm" data-testid={`item-${item.slot_id}`}>
      <div className="flex gap-4">
        <Thumb src={p.image_url} name={p.name} />
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-xs font-medium uppercase tracking-wide text-muted">{item.category_label}</span>
            {item.status === "confirmed" && <Badge tone="ok">{STATUS_LABEL.confirmed}</Badge>}
            {item.user_choice && <Badge tone="brand">Your choice</Badge>}
            {p.theme_matches && p.theme_matches.length > 0 && <Badge tone="brand" title="The listing's own title mentions your theme">Mentions {p.theme_matches.join(", ")}</Badge>}
            {p.simulated ? <Badge tone="bad" title="Invented for development; not a real product">Sample data</Badge>
              : <Badge tone="neutral" title={`Retrieved ${timeAgo(p.retrieved_at)} from Google Shopping`}>Real listing</Badge>}
          </div>
          <h4 className="mt-1 line-clamp-2 font-medium leading-snug" title={p.name}>{p.name}</h4>
          <div className="mt-1 flex flex-wrap items-baseline gap-x-3 text-sm text-muted">
            <span className="text-lg font-semibold tabular-nums text-ink">{inr(p.price)}</span>
            <span>{p.retailer ?? "Seller not stated"}</span>
            {p.rating !== null && <span>★ {p.rating}{p.review_count ? ` (${p.review_count})` : ""}</span>}
          </div>
          <div className="mt-2 flex items-center gap-3">
            <div className="flex-1"><ScoreBar value={item.utility} label={`Utility score for ${item.label}`} /></div>
            <span className="text-xs tabular-nums text-muted" title="Utility score out of 100">{item.utility?.toFixed(1)}/100</span>
          </div>
          <div className="mt-1 text-xs text-muted">
            Score uses {percent(item.completeness)} of the criteria
            {item.unavailable && item.unavailable.length > 0 && <> · not stated: {item.unavailable.join(", ")}</>}
          </div>
        </div>
      </div>

      {item.theme_unmatched && (
        <div className="mt-3 rounded-lg border border-warn/30 bg-warn-soft px-3 py-2 text-xs text-warn" data-testid="theme-note">
          <strong>Not themed:</strong> no saved listing mentions your theme, so this is a plain pick.
          {live?.enabled && item.search_query ? (
            <> <button type="button" disabled={busy} onClick={() => onSearch(item.category)}
              className="font-medium underline underline-offset-2 disabled:opacity-60">Search live for “{item.search_query}” (1 request)</button></>
          ) : item.search_query ? <> A themed search would be “{item.search_query}”.</> : null}
        </div>
      )}

      <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
        <PurchaseLink product={p} />
        <div className="flex flex-wrap gap-1.5">
          <Button variant="primary" disabled={busy || inCart} onClick={() => onAddToCart(p.product_id)} aria-label={inCart ? "In cart" : "Add to cart"}>
            {inCart ? "✓ In cart" : "Add to cart"}
          </Button>
          <Button onClick={() => onShortlist(p.product_id)} aria-pressed={starred} aria-label={starred ? "Remove from shortlist" : "Add to shortlist"}>
            {starred ? "★ Shortlisted" : "☆ Shortlist"}
          </Button>
          <Button onClick={() => setOpen(open === "why" ? "" : "why")} aria-expanded={open === "why"}>Why this?</Button>
          <Button onClick={() => setOpen(open === "alts" ? "" : "alts")} aria-expanded={open === "alts"}>
            Alternatives ({item.alternatives.length})
          </Button>
        </div>
      </div>

      {open === "why" && (
        <div className="mt-3 rounded-lg bg-subtle p-3 text-sm">
          <p>{item.why}</p>
          {item.contributions && Object.keys(item.contributions).length > 0 && (
            <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-xs sm:grid-cols-3">
              {Object.entries(item.contributions).map(([k, v]) => (
                <div key={k} className="flex justify-between"><dt className="text-muted">{COMPONENT_LABEL[k] ?? k}</dt><dd className="tabular-nums">{v.toFixed(1)} pts</dd></div>
              ))}
            </dl>
          )}
          <p className="mt-2 text-xs text-muted">
            {item.inspected} listing(s) checked{item.dropped_accessories ? `, ${item.dropped_accessories} dropped as accessories` : ""}.
            Budget ceiling for this item: {inr(item.ceiling)}.
          </p>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {live?.enabled && (
              <Button disabled={busy} onClick={() => onSearch(item.category)} aria-label={`Refresh ${item.category_label} listings`}>
                Refresh listings (1 request)
              </Button>
            )}
            <Button variant="danger" disabled={busy} onClick={() => onOutOfStock(p.product_id)}>Mark out of stock</Button>
            <Button variant="danger" disabled={busy} onClick={() => onRemove(item.slot_id)}>Remove item</Button>
            {item.user_choice && <Button disabled={busy} onClick={() => onClearChoice(item.slot_id)}>Back to SmartNest's pick</Button>}
          </div>
        </div>
      )}

      {open === "alts" && (
        <ul className="mt-3 divide-y divide-line rounded-lg border border-line px-3">
          {item.alternatives.length === 0 && <li className="py-3 text-sm text-muted">No other option fits this item's budget.</li>}
          {item.alternatives.map((a) => <AlternativeRow key={a.product_id} alt={a} slotId={item.slot_id} busy={busy} inCart={cartIds.includes(a.product_id)} onChoose={onChoose} onAddToCart={onAddToCart} />)}
        </ul>
      )}
    </article>
  );
}
