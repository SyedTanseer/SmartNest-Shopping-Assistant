import { Suspense, lazy, useCallback, useEffect, useState } from "react";
import { ApiError, api } from "./api";
import { AgentsView } from "./components/AgentsView";
import { ChatPanel } from "./components/ChatPanel";
import { ThemeToggle } from "./components/ThemeToggle";
import { CartView } from "./components/CartView";
import { CompareView } from "./components/CompareView";
import { PlanView } from "./components/PlanView";
import { timeAgo } from "./format";
import type { CatalogInfo, EventBody, Health, SessionView } from "./types";
import { Badge, Button } from "./ui";

// Recharts is large; load it only when the Budget tab is opened.
const BudgetView = lazy(() => import("./components/BudgetView").then((m) => ({ default: m.BudgetView })));

const TABS = [
  { id: "plan", label: "Plan" },
  { id: "budget", label: "Budget" },
  { id: "compare", label: "Compare" },
  { id: "cart", label: "Cart" },
  { id: "agents", label: "Agents" },
] as const;
type Tab = (typeof TABS)[number]["id"];

const STORE_KEY = "smartnest.session";
const remember = (id: string | null) => {
  try { id ? localStorage.setItem(STORE_KEY, id) : localStorage.removeItem(STORE_KEY); } catch { /* private mode */ }
};
const recall = (): string | null => {
  try { return localStorage.getItem(STORE_KEY); } catch { return null; }
};

export default function App() {
  const [session, setSession] = useState<SessionView | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [catalog, setCatalog] = useState<CatalogInfo | null>(null);
  const [tab, setTab] = useState<Tab>("plan");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pulse, setPulse] = useState(0);

  const fail = (e: unknown) => setError(e instanceof ApiError || e instanceof Error ? e.message : "Something went wrong");

  // Boot: server status, then the remembered session (or a fresh one).
  useEffect(() => {
    let live = true;
    (async () => {
      try {
        const [h, c] = await Promise.all([api.health(), api.catalog()]);
        if (!live) return;
        setHealth(h); setCatalog(c);
        const saved = recall();
        if (saved) {
          try { const s = await api.getSession(saved); if (live) { setSession(s); return; } }
          catch (e) { if (!(e instanceof ApiError && e.status === 404)) throw e; }
        }
        const fresh = await api.createSession();
        if (live) { setSession(fresh); remember(fresh.id); }
      } catch (e) { if (live) fail(e); }
    })();
    return () => { live = false; };
  }, []);

  const guarded = useCallback(async (job: () => Promise<void>) => {
    setBusy(true); setError(null);
    try { await job(); setPulse((n) => n + 1); } catch (e) { fail(e); } finally { setBusy(false); }
  }, []);

  const sid = session?.id;
  const send = (text: string) => sid && guarded(async () => { setSession((await api.chat(sid, text)).session); });
  const event = (body: EventBody) => sid && guarded(async () => { setSession(await api.event(sid, body)); });
  const toggleShortlist = (productId: string) => sid && guarded(async () => {
    await api.shortlist(sid, productId);
    setSession(await api.getSession(sid));
  });
  const searchLive = (category: string) => {
    if (!sid) return;
    const left = health?.live_search?.remaining;
    const ok = window.confirm(`Search the live shopping API for this category? This uses 1 SerpApi request${left != null ? ` (${left} left)` : ""}.`);
    if (!ok) return;
    void guarded(async () => {
      const r = await api.search(sid, category);
      setSession(r.session);
      setHealth(await api.health());
    });
  };
  const searchMany = (categories: string[]) => {
    if (!sid || categories.length === 0) return;
    const left = health?.live_search?.remaining;
    const ok = window.confirm(`Search the live shopping API for ${categories.length} categories? This uses ${categories.length} SerpApi requests${left != null ? ` (${left} left)` : ""}.`);
    if (!ok) return;
    void guarded(async () => {
      for (const category of categories) {            // one at a time, so a failure stops the run and nothing more is spent
        const r = await api.search(sid, category);
        setSession(r.session);
      }
      setHealth(await api.health());
    });
  };
  const addToCart = (product_id: string) => event({ type: "cart_add", product_id });
  const newSession = () => guarded(async () => {
    const fresh = await api.createSession();
    setSession(fresh); remember(fresh.id); setTab("plan");
  });

  const plan = session?.plan ?? null;
  // Follow-up chips that fit THIS plan (a child's room has no sofa), always including budget and cart prompts.
  const firstPick = plan?.items.find((i) => i.product);
  const firstRoom = session?.rooms[0]?.label;
  const chips = plan ? [
    ...(firstPick ? [`Why this ${firstPick.category_label.toLowerCase()}?`] : []),
    "How much is left?",
    ...(firstRoom ? [`Show ${firstRoom.toLowerCase()}`] : []),
    ...(firstPick ? [`I don't need a ${firstPick.category_label.toLowerCase()}`] : []),
    `Increase the budget to ${Math.round((plan.request.total_budget * 1.5) / 1000) * 1000}`,
    "Add everything to my cart",
  ] : [];
  const cartIds = (session?.cart?.lines ?? []).map((l) => l.product.product_id);
  const origins = catalog?.categories ?? [];
  const real = origins.filter((c) => c.origin === "serpapi");
  const oldest = real.map((c) => c.retrieved_at).filter(Boolean).sort()[0] ?? null;

  return (
    <div className="flex min-h-full flex-col">
      <header className="border-b border-line bg-surface">
        <div className="mx-auto flex max-w-[1500px] flex-wrap items-center justify-between gap-3 px-4 py-3">
          <div className="flex items-center gap-2.5">
            <svg width="28" height="28" viewBox="0 0 32 32" aria-hidden><rect width="32" height="32" rx="8" fill="#0f766e" />
              <path d="M6 16 16 7l10 9v9H6z" fill="none" stroke="#fff" strokeWidth="2.4" strokeLinejoin="round" /></svg>
            <div><h1 className="text-lg font-semibold leading-none">SmartNest</h1>
              <p className="text-xs text-muted">Autonomous home furnishing planner</p></div>
          </div>
          <div className="flex flex-wrap items-center gap-1.5" aria-label="System status">
            {real.length > 0
              ? <Badge tone="ok" title={`${real.length} of ${origins.length} categories use real Google Shopping listings`}>Real data · {real.length} categories · {timeAgo(oldest)}</Badge>
              : origins.length > 0 && <Badge tone="bad" title="No real snapshots found; listings are invented samples">Sample data only</Badge>}
            {health && <Badge tone={health.gemini ? "brand" : "neutral"}>{health.gemini ? "Gemini on" : "Gemini off"}</Badge>}
            {health && <Badge>{health.persistence === "mongodb" ? "MongoDB" : "In-memory"}</Badge>}
            <Button onClick={newSession} disabled={busy}>New plan</Button>
            <ThemeToggle />
          </div>
        </div>
      </header>

      {error && (
        <div role="alert" className="border-b border-bad/30 bg-bad-soft px-4 py-2 text-sm text-bad">
          <div className="mx-auto flex max-w-[1500px] items-center justify-between gap-3">
            <span>{error}</span><button className="underline" onClick={() => setError(null)}>Dismiss</button>
          </div>
        </div>
      )}

      <main className="mx-auto grid w-full max-w-[1500px] flex-1 gap-4 p-4 lg:grid-cols-[380px_1fr]">
        <aside className="flex h-[32rem] min-h-0 flex-col overflow-hidden rounded-xl border border-line bg-paper shadow-sm lg:sticky lg:top-4 lg:h-[calc(100vh-7.5rem)]"
          aria-label="Assistant">
          {session
            ? <ChatPanel history={session.history} hasPlan={session.has_plan} busy={busy} onSend={send} suggestions={chips} />
            : <div className="p-4 text-sm text-muted">Connecting…</div>}
        </aside>

        <section aria-label="Workspace" className="min-w-0">
          <div role="tablist" aria-label="Views" className="scroll-thin mb-4 flex gap-1 overflow-x-auto border-b border-line">
            {TABS.map((t) => (
              <button key={t.id} role="tab" id={`tab-${t.id}`} aria-selected={tab === t.id} aria-controls={`panel-${t.id}`}
                onClick={() => setTab(t.id)}
                className={`-mb-px shrink-0 border-b-2 px-3 py-2 text-sm font-medium sm:px-4 ${tab === t.id ? "border-brand text-brand-dark" : "border-transparent text-muted hover:text-ink"}`}>
                {t.label}
                {t.id === "cart" && session?.cart && session.cart.count > 0 && <span className="ml-1.5 rounded-full bg-brand-soft px-1.5 text-xs">{session.cart.count}</span>}
                {t.id === "compare" && session && session.shortlist.length > 0 && <span className="ml-1.5 rounded-full bg-brand-soft px-1.5 text-xs">{session.shortlist.length}</span>}
              </button>
            ))}
          </div>
          <div role="tabpanel" id={`panel-${tab}`} aria-labelledby={`tab-${tab}`}>
            {tab === "plan" && (
              <PlanView rooms={session?.rooms ?? []} shortlist={session?.shortlist ?? []} busy={busy}
                cartIds={cartIds} onAddToCart={addToCart} live={health?.live_search} onSearch={searchLive} onSearchMany={searchMany}
                budgetNeeded={plan?.summary.budget_needed}
                onRaiseBudget={(total_budget) => event({ type: "budget_change", total_budget })}
                notes={plan?.data_notes ?? []} assumptions={plan?.assumptions ?? []}
                onShortlist={toggleShortlist}
                onChoose={(slot_id, product_id) => event({ type: "choose_product", slot_id, product_id })}
                onClearChoice={(slot_id) => event({ type: "clear_choice", slot_id })}
                onOutOfStock={(product_id) => event({ type: "out_of_stock", product_id })}
                onRemove={(slot_id) => event({ type: "remove_item", slot_id })} />
            )}
            {tab === "budget" && (
              <Suspense fallback={<p className="text-sm text-muted" role="status">Loading charts…</p>}>
                <BudgetView budget={session?.budget ?? null} />
              </Suspense>
            )}
            {tab === "compare" && sid && (
              <CompareView sessionId={sid} items={plan?.items ?? []} shortlist={session?.shortlist ?? []} onToggle={toggleShortlist}
                cartIds={cartIds} onAddToCart={addToCart} />
            )}
            {tab === "cart" && (
              <CartView cart={session?.cart ?? null} busy={busy}
                onSetQty={(product_id, qty) => event({ type: "cart_set", product_id, qty })}
                onRemove={(product_id) => event({ type: "cart_remove", product_id })}
                onClear={() => event({ type: "cart_clear" })}
                onAddPlan={() => event({ type: "cart_add_plan" })} />
            )}
            {tab === "agents" && sid && <AgentsView sessionId={sid} refreshKey={pulse} />}
          </div>
        </section>
      </main>
    </div>
  );
}
