// Shapes returned by the SmartNest API (backend/src/smartnest/api).

export type ItemStatus = "confirmed" | "unverified" | "unfilled";

export interface Product {
  product_id: string;
  name: string;
  price: number;
  retailer: string | null;
  image_url: string | null;
  data_origin: "serpapi" | "simulated";
  simulated: boolean;
  buy_url: string | null;
  link_label: string;
  url_type: string | null;
  rating: number | null;
  review_count: number | null;
  seating_capacity: number | null;
  theme_matches?: string[];
  retrieved_at: string | null;
  availability: string | null;
}

export interface Alternative extends Product {
  state: "confirmed" | "unverified";
  utility: number | null;
  completeness: number | null;
  gaps: string[];
}

export interface PlanItem {
  slot_id: string;
  room_id: string;
  room: string;
  category: string;
  category_label: string;
  label: string;
  ceiling: number;
  status: ItemStatus;
  user_choice?: boolean;
  product: Product | null;
  utility: number | null;
  completeness: number | null;
  unavailable?: string[];
  gaps?: string[];
  fit?: string | null;
  components?: Record<string, number>;
  weights?: Record<string, number>;
  contributions?: Record<string, number>;
  alternatives: Alternative[];
  why: string;
  no_data?: boolean;
  theme_unmatched?: boolean;
  cheapest_price?: number | null;
  search_query?: string;
  inspected: number;
  dropped_accessories: number;
  data_origin: string;
}

export interface Summary {
  total_budget: number;
  spent: number;
  remaining: number;
  utilisation: number;
  within_budget: boolean;
  by_room: Record<string, number>;
  by_category: Record<string, number>;
  unfilled: string[];
  confirmed: number;
  unverified: number;
  unfilled_count: number;
  no_data_count?: number;
  minimum_total?: number | null;
  shortfall?: number;
  budget_needed?: number | null;
  themes?: string[];
  themed_items?: number;
  theme_unmatched_count?: number;
  items: number;
}

export interface Plan {
  request: {
    total_budget: number; bedrooms: number | null; style: string | null; colour: string | null; label: string;
    scope: string[]; items?: string[]; audience: string | null; age: number | null; themes: string[];
  };
  version: number;
  updated_at: string;
  assumptions: string[];
  items: PlanItem[];
  summary: Summary;
  data_notes: string[];
}

export interface RoomView {
  room_id: string;
  label: string;
  spent: number;
  allocated: number;
  filled: number;
  count: number;
  items: PlanItem[];
}

export interface BudgetView {
  total: number;
  spent: number;
  remaining: number;
  utilisation: number;
  within_budget: boolean;
  by_room: { name: string; value: number }[];
  by_category: { name: string; value: number }[];
  items: { label: string; room: string; category: string; ceiling: number; price: number | null; status: ItemStatus }[];
  allocation_vs_spend: { room: string; allocated: number; spent: number }[];
  counts: { confirmed: number; unverified: number; unfilled: number };
}

export interface ChatMessage {
  role: "user" | "assistant";
  text: string;
  kind: string;
  time: string;
}

export interface Transition {
  time: string;
  slot_id: string;
  label: string;
  kind: string;
  before: string | null;
  after: string | null;
  reason: string;
}

export interface CartLine {
  product: Product;
  qty: number;
  line_total: number;
  category: string | null;
  status: "confirmed" | "unverified";
  in_plan: boolean;
  out_of_stock: boolean;
}

export interface Cart {
  lines: CartLine[];
  count: number;
  total: number;
  budget: number;
  remaining: number;
  within_budget: boolean;
  unverified: number;
  warnings: string[];
}

export interface SessionView {
  id: string;
  created_at: string;
  has_plan: boolean;
  plan: Plan | null;
  rooms: RoomView[];
  budget: BudgetView | null;
  shortlist: string[];
  cart: Cart | null;
  transitions: Transition[];
  history: ChatMessage[];
  draft: Record<string, unknown>;
}

export interface Reply {
  text: string;
  kind: string;
  changed: boolean;
  needs: string[];
  wording: "rules" | "gemini";
}

export interface ActivityEvent {
  seq: number;
  time: string;
  agent: string;
  kind: "message" | "decision" | "info" | "warning" | "replan";
  summary: string;
  peer: string | null;
  task: string | null;
  verified: boolean | null;
  duration_ms: number | null;
  details: Record<string, unknown>;
}

export interface AgentInfo {
  id: string;
  role: "coordinator" | "worker";
  actions: number;
  requests_received: number;
  avg_ms: number | null;
  fingerprint: string | null;
}

export interface Activity {
  agents: AgentInfo[];
  security: { accepted: number; rejected: number; by_code: Record<string, number> };
  messages: number;
  unverified_messages: number;
  total_events: number;
  events: ActivityEvent[];
}

export interface Health {
  status: string;
  gemini: boolean;
  persistence: string;
  serpapi_quota: { used: number; limit: number | null } | null;
  sessions: number;
  catalog: { simulated_allowed: boolean };
  live_search?: LiveStatus;
}

export interface LiveStatus {
  enabled: boolean;
  remaining?: number | null;
  limit?: number | null;
  used?: number;
}

export interface CatalogInfo {
  categories: { category: string; label: string; origin: string; listings: number; retrieved_at: string | null }[];
}

export interface OptionRow {
  product_id: string;
  name: string;
  price: number;
  retailer: string | null;
  rating: number | null;
  availability: string | null;
  fits_budget: boolean;
  is_pick: boolean;
}

export interface Options {
  slot_id: string;
  label: string;
  budget_available: number;
  options: OptionRow[];
}

export interface CompareProduct extends Alternative {
  item: string;
  slot_id: string;
  is_pick: boolean;
  contributions: Record<string, number> | null;
}

export type EventBody =
  | { type: "budget_change"; total_budget: number }
  | { type: "out_of_stock"; product_id: string }
  | { type: "remove_item"; slot_id: string }
  | { type: "choose_product"; slot_id: string; product_id: string }
  | { type: "clear_choice"; slot_id: string }
  | { type: "cart_add"; product_id: string; qty?: number }
  | { type: "cart_set"; product_id: string; qty: number }
  | { type: "cart_remove"; product_id: string }
  | { type: "cart_clear" }
  | { type: "cart_add_plan" }
  | { type: "preference_change"; style?: string | null; colour?: string | null };
