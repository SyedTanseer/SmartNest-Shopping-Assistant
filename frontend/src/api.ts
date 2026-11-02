import type {
  Activity, CatalogInfo, CompareProduct, EventBody, Health, Options, Reply, SessionView,
} from "./types";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...init });
  } catch {
    throw new ApiError(0, "Cannot reach the SmartNest server. Is the backend running?");
  }
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (typeof body?.detail === "string") detail = body.detail;
      else if (Array.isArray(body?.detail) && body.detail[0]?.msg) detail = String(body.detail[0].msg);
    } catch { /* keep the generic message */ }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

const post = (body: unknown): RequestInit => ({ method: "POST", body: JSON.stringify(body) });
const sid = (id: string) => `/api/sessions/${encodeURIComponent(id)}`;

export const api = {
  health: () => call<Health>("/api/health"),
  catalog: () => call<CatalogInfo>("/api/catalog"),
  createSession: (body: { message?: string; total_budget?: number; bedrooms?: number } = {}) =>
    call<SessionView>("/api/sessions", post(body)),
  getSession: (id: string) => call<SessionView>(sid(id)),
  chat: (id: string, message: string) =>
    call<{ reply: Reply; session: SessionView }>(`${sid(id)}/messages`, post({ message })),
  event: (id: string, event: EventBody) => call<SessionView>(`${sid(id)}/events`, post(event)),
  options: (id: string, slot: string) =>
    call<Options>(`${sid(id)}/items/${encodeURIComponent(slot)}/options`),
  shortlist: (id: string, productId: string) =>
    call<{ shortlist: string[] }>(`${sid(id)}/shortlist`, post({ product_id: productId })),
  compare: (id: string, productIds: string[]) =>
    call<{ products: CompareProduct[] }>(`${sid(id)}/compare`, post({ product_ids: productIds })),
  search: (id: string, category: string) =>
    call<{ found: number; query: string; session: SessionView }>(`${sid(id)}/search`, post({ category })),
  activity: (id: string, since = 0) => call<Activity>(`${sid(id)}/activity?since=${since}`),
};
