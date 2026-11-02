import { Bar, BarChart, CartesianGrid, Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { inr, inrShort, percent } from "../format";
import { useChartColors } from "../theme";
import type { BudgetView as Budget } from "../types";
import { Badge, Card, Empty, Stat } from "../ui";


const tip = (v: unknown) => inr(typeof v === "number" ? v : Number(v));

export function BudgetView({ budget }: { budget: Budget | null }) {
  const C = useChartColors();
  if (!budget) return <Empty title="No budget yet">Once a plan exists you will see where every rupee goes.</Empty>;
  const over = budget.spent > budget.total;
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Total budget" value={inr(budget.total)} />
        <Stat label="Planned spend" value={inr(budget.spent)} hint={`${percent(budget.utilisation)} of budget`} tone={over ? "bad" : undefined} />
        <Stat label="Remaining" value={inr(budget.remaining)} tone={over ? "bad" : "ok"} />
        <Stat label="Items" value={`${budget.counts.confirmed + budget.counts.unverified}/${budget.items.length}`}
          hint={`${budget.counts.unfilled} unfilled`} />
      </div>

      <div className="h-3 overflow-hidden rounded-full bg-track" role="meter" aria-label="Budget used"
        aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(budget.utilisation * 100)}>
        <div className={`h-full ${over ? "bg-bad" : "bg-fill"}`} style={{ width: `${Math.min(100, budget.utilisation * 100)}%` }} />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card className="p-4">
          <h3 className="font-semibold">Spend by room</h3>
          <div className="h-64" aria-hidden>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={budget.by_room} dataKey="value" nameKey="name" innerRadius={55} outerRadius={85} paddingAngle={2} stroke={C.surface} strokeWidth={2}>
                  {budget.by_room.map((_, i) => <Cell key={i} fill={C.palette[i % C.palette.length]} />)}
                </Pie>
                <Tooltip formatter={tip} contentStyle={{ background: C.surface, border: `1px solid ${C.grid}`, borderRadius: 8, color: C.ink }}
                  itemStyle={{ color: C.ink }} labelStyle={{ color: C.text }} cursor={{ fill: C.grid, opacity: 0.35 }} />
                <Legend formatter={(v) => <span style={{ color: C.text }}>{v}</span>} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card className="p-4">
          <h3 className="font-semibold">Spend by category</h3>
          <div style={{ height: Math.max(256, budget.by_category.length * 30 + 40) }} aria-hidden>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={budget.by_category} layout="vertical" margin={{ left: 24 }}>
                <CartesianGrid horizontal={false} stroke={C.grid} />
                <XAxis type="number" tickFormatter={inrShort} tick={{ fill: C.text }} stroke={C.grid} />
                <YAxis type="category" dataKey="name" width={96} interval={0} tick={{ fill: C.text }} stroke={C.grid} />
                <Tooltip formatter={tip} contentStyle={{ background: C.surface, border: `1px solid ${C.grid}`, borderRadius: 8, color: C.ink }}
                  itemStyle={{ color: C.ink }} labelStyle={{ color: C.text }} cursor={{ fill: C.grid, opacity: 0.35 }} />
                <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                  {budget.by_category.map((_, i) => <Cell key={i} fill={C.palette[i % C.palette.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      <Card className="p-4">
        <h3 className="font-semibold">Allocated vs spent, per room</h3>
        <p className="text-sm text-muted">Each item gets a hard price ceiling from the budget. Unspent allocation stays in your pocket.</p>
        <div className="mt-2 h-64" aria-hidden>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={budget.allocation_vs_spend}>
              <CartesianGrid vertical={false} stroke={C.grid} />
              <XAxis dataKey="room" tick={{ fill: C.text }} stroke={C.grid} />
              <YAxis tickFormatter={inrShort} tick={{ fill: C.text }} stroke={C.grid} />
              <Tooltip formatter={tip} contentStyle={{ background: C.surface, border: `1px solid ${C.grid}`, borderRadius: 8, color: C.ink }}
                  itemStyle={{ color: C.ink }} labelStyle={{ color: C.text }} cursor={{ fill: C.grid, opacity: 0.35 }} />
              <Legend formatter={(v) => <span style={{ color: C.text }}>{v}</span>} />
              <Bar dataKey="allocated" name="Allocated" fill={C.allocated} radius={[4, 4, 0, 0]} />
              <Bar dataKey="spent" name="Spent" fill={C.spent} radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Card>

      <Card className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <caption className="p-4 pb-2 text-left font-semibold">Every item</caption>
          <thead className="text-xs uppercase tracking-wide text-muted">
            <tr><th className="px-4 py-2">Item</th><th className="px-4 py-2 text-right">Ceiling</th><th className="px-4 py-2 text-right">Price</th><th className="px-4 py-2">Status</th></tr>
          </thead>
          <tbody>
            {budget.items.map((row) => (
              <tr key={row.label} className="border-t border-line">
                <td className="px-4 py-2">{row.label}</td>
                <td className="px-4 py-2 text-right tabular-nums text-muted">{inr(row.ceiling)}</td>
                <td className="px-4 py-2 text-right tabular-nums">{inr(row.price)}</td>
                <td className="px-4 py-2">{row.status === "unfilled" ? <Badge tone="bad">unfilled</Badge> : <span className="text-muted">Picked</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  );
}
