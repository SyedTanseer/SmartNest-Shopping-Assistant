import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { Activity, ActivityEvent } from "../types";
import { Badge, Card, Empty, Stat } from "../ui";

const KIND_TONE = { message: "brand", decision: "ok", info: "neutral", warning: "bad", replan: "warn" } as const;
const FILTERS = ["all", "message", "decision", "replan", "warning"] as const;
const LABEL: Record<string, string> = {
  coordinator: "Coordinator", space_planning: "Space Planning", product_research: "Product Research",
  comparison: "Comparison", budget: "Budget",
};

/** Polls the activity log while the tab is open; `refreshKey` forces a refresh after the plan changes. */
export function AgentsView({ sessionId, refreshKey }: { sessionId: string; refreshKey: number }) {
  const [data, setData] = useState<Activity | null>(null);
  const [events, setEvents] = useState<ActivityEvent[]>([]);
  const [filter, setFilter] = useState<(typeof FILTERS)[number]>("all");
  const [error, setError] = useState<string | null>(null);
  const last = useRef(0);

  useEffect(() => { last.current = 0; setEvents([]); setData(null); }, [sessionId]);

  useEffect(() => {
    let live = true;
    const pull = () =>
      api.activity(sessionId, last.current)
        .then((a) => {
          if (!live) return;
          setData(a);
          setError(null);
          if (a.events.length) {
            last.current = a.events[a.events.length - 1].seq;
            setEvents((prev) => [...prev, ...a.events].slice(-500));
          }
        })
        .catch((e: Error) => live && setError(e.message));
    pull();
    const timer = setInterval(pull, 3000);
    return () => { live = false; clearInterval(timer); };
  }, [sessionId, refreshKey]);

  if (error && !data) return <p role="alert" className="text-sm text-bad">{error}</p>;
  if (!data) return <Empty title="Loading agent activity…" />;
  const shown = events.filter((e) => filter === "all" || e.kind === filter).slice().reverse();

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Agents" value={data.agents.length} />
        <Stat label="Signed messages" value={data.messages} hint="each Ed25519-signed" />
        <Stat label="Failed signature checks" value={data.unverified_messages} tone={data.unverified_messages ? "bad" : "ok"} />
        <Stat label="Router decisions" value={`${data.security.accepted} ok / ${data.security.rejected} refused`} />
      </div>

      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        {data.agents.map((a) => (
          <Card key={a.id} className="p-4" >
            <div className="flex items-center justify-between">
              <h3 className="font-semibold">{LABEL[a.id] ?? a.id}</h3>
              <Badge tone={a.role === "coordinator" ? "brand" : "neutral"}>{a.role}</Badge>
            </div>
            <dl className="mt-2 grid grid-cols-3 gap-2 text-xs">
              <div><dt className="text-muted">Actions</dt><dd className="text-base font-semibold tabular-nums">{a.actions}</dd></div>
              <div><dt className="text-muted">Requests</dt><dd className="text-base font-semibold tabular-nums">{a.requests_received}</dd></div>
              <div><dt className="text-muted">Avg reply</dt><dd className="text-base font-semibold tabular-nums">{a.avg_ms !== null ? `${a.avg_ms} ms` : "—"}</dd></div>
            </dl>
            <div className="mt-2 text-xs text-muted">Key fingerprint <code className="rounded bg-subtle px-1">{a.fingerprint}</code></div>
          </Card>
        ))}
      </div>

      <Card className="p-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h3 className="font-semibold">Activity timeline</h3>
          <div role="group" aria-label="Filter events" className="flex gap-1">
            {FILTERS.map((f) => (
              <button key={f} onClick={() => setFilter(f)} aria-pressed={filter === f}
                className={`rounded-full border px-2.5 py-0.5 text-xs ${filter === f ? "border-fill bg-fill text-white" : "border-line bg-surface"}`}>{f}</button>
            ))}
          </div>
        </div>
        <ol className="scroll-thin mt-3 max-h-[28rem] space-y-1.5 overflow-y-auto pr-1">
          {shown.length === 0 && <li className="text-sm text-muted">Nothing yet.</li>}
          {shown.map((e) => (
            <li key={e.seq} className="flex gap-2 rounded-lg border border-line bg-surface px-3 py-2 text-sm">
              <time className="shrink-0 pt-0.5 text-xs tabular-nums text-muted" dateTime={e.time}>{e.time.slice(11, 19)}</time>
              <Badge tone={KIND_TONE[e.kind]}>{e.kind}</Badge>
              <div className="min-w-0 flex-1">
                <span className="font-medium">{LABEL[e.agent] ?? e.agent}</span>
                {e.peer && <span className="text-muted"> → {LABEL[e.peer] ?? e.peer}</span>}
                <span> — {e.summary}</span>
                {e.kind === "message" && (
                  <span className="ml-1 text-xs text-muted">
                    {e.verified ? "✓ signature verified" : "✗ signature failed"}{e.duration_ms !== null && ` · ${e.duration_ms} ms`}
                  </span>
                )}
              </div>
            </li>
          ))}
        </ol>
      </Card>
    </div>
  );
}
