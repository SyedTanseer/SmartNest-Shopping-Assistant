import { useState } from "react";
import { inr } from "../format";
import type { Cart, CartLine } from "../types";
import { Badge, Button, Card, Empty, Stat } from "../ui";
import { PurchaseLink } from "./ItemCard";

interface Props {
  cart: Cart | null;
  busy: boolean;
  onSetQty: (productId: string, qty: number) => void;
  onRemove: (productId: string) => void;
  onClear: () => void;
  onAddPlan: () => void;
}

/** The buy list as plain text, so it can be pasted into a note or message. */
export function cartText(cart: Cart): string {
  const rows = cart.lines.map((l) =>
    `${l.qty} x ${l.product.name} - ${inr(l.line_total)}${l.product.simulated || !l.product.buy_url ? " (sample, no link)" : ` - ${l.product.buy_url}`}`);
  return [...rows, `Total: ${inr(cart.total)} of ${inr(cart.budget)} budget`].join("\n");
}

function Line({ line, busy, onSetQty, onRemove }: { line: CartLine; busy: boolean } & Pick<Props, "onSetQty" | "onRemove">) {
  const p = line.product;
  return (
    <li className="flex flex-wrap items-center gap-x-4 gap-y-2 py-3" data-testid={`cart-${p.product_id}`}>
      <div className="min-w-0 flex-1 basis-64">
        <div className="flex flex-wrap items-center gap-1.5">
          {line.status === "confirmed" && <Badge tone="ok">Confirmed in stock</Badge>}
          {line.out_of_stock && <Badge tone="bad">Now out of stock</Badge>}
          {p.simulated && <Badge tone="bad">Sample data</Badge>}
          {!line.in_plan && <Badge>Not a plan pick</Badge>}
        </div>
        <div className="mt-1 line-clamp-2 font-medium leading-snug" title={p.name}>{p.name}</div>
        <div className="text-xs text-muted">{p.retailer ?? "Seller not stated"} · {inr(p.price)} each</div>
        <div className="mt-1"><PurchaseLink product={p} /></div>
      </div>
      <div className="flex items-center gap-1" role="group" aria-label={`Quantity of ${p.name}`}>
        <Button disabled={busy} onClick={() => onSetQty(p.product_id, line.qty - 1)} aria-label="Decrease quantity">−</Button>
        <span className="w-8 text-center tabular-nums" aria-live="polite">{line.qty}</span>
        <Button disabled={busy || line.qty >= 10} onClick={() => onSetQty(p.product_id, line.qty + 1)} aria-label="Increase quantity">+</Button>
      </div>
      <div className="w-24 text-right font-semibold tabular-nums">{inr(line.line_total)}</div>
      <Button variant="danger" disabled={busy} onClick={() => onRemove(p.product_id)} aria-label={`Remove ${p.name} from cart`}>Remove</Button>
    </li>
  );
}

export function CartView({ cart, busy, onSetQty, onRemove, onClear, onAddPlan }: Props) {
  const [copied, setCopied] = useState(false);
  if (!cart) return <Empty title="No cart yet">Once there is a plan you can add products to your cart.</Empty>;

  const copy = async () => {
    try { await navigator.clipboard.writeText(cartText(cart)); setCopied(true); setTimeout(() => setCopied(false), 2000); }
    catch { /* clipboard unavailable */ }
  };
  const used = cart.budget ? Math.min(100, (cart.total / cart.budget) * 100) : 0;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-3">
        <Stat label="Cart total" value={inr(cart.total)} tone={cart.within_budget ? undefined : "bad"} />
        <Stat label="Budget left" value={inr(cart.remaining)} tone={cart.within_budget ? "ok" : "bad"} />
        <Stat label="Products" value={cart.count} hint={`${cart.lines.length} line(s)`} />
      </div>
      <div className="h-2.5 overflow-hidden rounded-full bg-track" role="meter" aria-label="Cart against budget"
        aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(used)}>
        <div className={`h-full ${cart.within_budget ? "bg-fill" : "bg-bad"}`} style={{ width: `${used}%` }} />
      </div>

      {cart.warnings.map((w) => <p key={w} role="alert" className="rounded-lg border border-bad/30 bg-bad-soft px-3 py-2 text-sm text-bad">{w}</p>)}

      <div className="flex flex-wrap gap-2">
        <Button variant="primary" disabled={busy} onClick={onAddPlan}>Add all plan picks</Button>
        <Button disabled={cart.lines.length === 0} onClick={copy}>{copied ? "Copied ✓" : "Copy buy list"}</Button>
        <Button variant="danger" disabled={busy || cart.lines.length === 0} onClick={onClear}>Clear cart</Button>
      </div>

      {cart.lines.length === 0 ? (
        <Empty title="Your cart is empty">Press “Add to cart” on a product, or add all of SmartNest's picks at once.</Empty>
      ) : (
        <Card className="px-4">
          <ul className="divide-y divide-line">
            {cart.lines.map((l) => <Line key={l.product.product_id} line={l} busy={busy} onSetQty={onSetQty} onRemove={onRemove} />)}
          </ul>
        </Card>
      )}
      <p className="text-xs text-muted">
        SmartNest does not sell anything. Each link opens the retailer's page (or Google Shopping, labelled), where you complete the purchase.
        The cart can never go over your total budget.
      </p>
    </div>
  );
}
