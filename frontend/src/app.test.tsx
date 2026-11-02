import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, api } from "./api";
import { BudgetView } from "./components/BudgetView";
import { ChatPanel } from "./components/ChatPanel";
import { ItemCard } from "./components/ItemCard";
import { inr, inrShort, percent, timeAgo } from "./format";
import type { BudgetView as Budget, PlanItem, Product } from "./types";

const product = (over: Partial<Product> = {}): Product => ({
  product_id: "P1", name: "Oak Sofa", price: 18628, retailer: "Wakefit", image_url: null, data_origin: "serpapi",
  simulated: false, buy_url: "https://www.wakefit.co/sofa", link_label: "Retailer page", url_type: "retailer",
  rating: 4.5, review_count: 12, seating_capacity: 3, retrieved_at: "2026-10-31T00:00:00Z", availability: "in_stock", ...over,
});

const item = (over: Partial<PlanItem> = {}): PlanItem => ({
  slot_id: "living_room.sofa", room_id: "living_room", room: "Living room", category: "sofa", category_label: "Sofa",
  label: "Living room sofa", ceiling: 32000, status: "confirmed", product: product(), utility: 58.8, completeness: 0.43,
  unavailable: ["style"], contributions: { price: 22.8, quality: 36 }, alternatives: [], why: "Highest utility.",
  inspected: 10, dropped_accessories: 0, data_origin: "serpapi", ...over,
});

const noop = {
  onShortlist: vi.fn(), onChoose: vi.fn(), onClearChoice: vi.fn(), onOutOfStock: vi.fn(), onRemove: vi.fn(), onAddToCart: vi.fn(), onSearch: vi.fn(),
};

afterEach(() => { vi.restoreAllMocks(); vi.clearAllMocks(); });

describe("format", () => {
  it("groups rupees the Indian way", () => {
    expect([0, 999, 1000, 200000, 1234567, -5000].map(inr)).toEqual(["₹0", "₹999", "₹1,000", "₹2,00,000", "₹12,34,567", "-₹5,000"]);
    expect(inr(null)).toBe("—");
    expect(inr(NaN)).toBe("—");
  });
  it("shortens axis values", () => {
    expect([500, 45000, 200000, 25000000].map(inrShort)).toEqual(["₹500", "₹45k", "₹2L", "₹2.5Cr"]);
  });
  it("formats percentages and ages", () => {
    expect(percent(0.43)).toBe("43%");
    expect(percent(null)).toBe("—");
    const now = new Date("2026-11-01T12:00:00Z");
    expect(timeAgo("2026-11-01T11:30:00Z", now)).toBe("under an hour ago");
    expect(timeAgo("2026-11-01T00:00:00Z", now)).toBe("12 h ago");
    expect(timeAgo("2026-10-20T12:00:00Z", now)).toBe("12 days ago");
    expect(timeAgo("garbage", now)).toBe("unknown");
    expect(timeAgo(null, now)).toBe("unknown");
  });
});

describe("ItemCard", () => {
  it("shows a confirmed real listing with a safe external link", () => {
    render(<ItemCard item={item()} shortlist={[]} cartIds={[]} busy={false} {...noop} />);
    expect(screen.getByText("Confirmed in stock")).toBeInTheDocument();
    expect(screen.getByText("Real listing")).toBeInTheDocument();
    expect(screen.getByText("₹18,628")).toBeInTheDocument();
    const link = screen.getByRole("link", { name: /Retailer page/ });
    expect(link).toHaveAttribute("href", "https://www.wakefit.co/sofa");
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
    expect(link).toHaveAttribute("target", "_blank");
  });

  it("shows no stock badge for unverified stock and never calls it confirmed", () => {
    render(<ItemCard item={item({ status: "unverified" })} shortlist={[]} cartIds={[]} busy={false} {...noop} />);
    expect(screen.queryByText("Stock unverified")).not.toBeInTheDocument();
    expect(screen.queryByText("Confirmed in stock")).not.toBeInTheDocument();
  });

  it("gives simulated products no purchase link and a sample badge", () => {
    const sim = product({ simulated: true, data_origin: "simulated", buy_url: null, link_label: "Sample listing - not a real product" });
    render(<ItemCard item={item({ product: sim })} shortlist={[]} cartIds={[]} busy={false} {...noop} />);
    expect(screen.getByText("Sample data")).toBeInTheDocument();
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
    expect(screen.getByText(/not a real product/)).toBeInTheDocument();
  });

  it("never renders a link for a simulated product even if a URL leaks in", () => {
    const sim = product({ simulated: true, buy_url: "https://evil.example" });
    render(<ItemCard item={item({ product: sim })} shortlist={[]} cartIds={[]} busy={false} {...noop} />);
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
  });

  it("explains an unfilled item instead of pretending", () => {
    render(<ItemCard item={item({ product: null, status: "unfilled", why: "No sofa fits; nothing was relaxed." })}
      shortlist={[]} cartIds={[]} busy={false} {...noop} />);
    expect(screen.getByText("Not filled")).toBeInTheDocument();
    expect(screen.getByText(/nothing was relaxed/)).toBeInTheDocument();
  });

  it("wires the actions", async () => {
    const user = userEvent.setup();
    const alt = { ...product({ product_id: "P2", name: "Pine Sofa", price: 20000 }), state: "unverified" as const, utility: 50, completeness: 0.4, gaps: [] };
    render(<ItemCard item={item({ alternatives: [alt] })} shortlist={[]} cartIds={[]} busy={false} {...noop} />);
    await user.click(screen.getByRole("button", { name: "Add to shortlist" }));
    expect(noop.onShortlist).toHaveBeenCalledWith("P1");
    await user.click(screen.getByRole("button", { name: /Alternatives \(1\)/ }));
    await user.click(screen.getByRole("button", { name: "Use Pine Sofa" }));
    expect(noop.onChoose).toHaveBeenCalledWith("living_room.sofa", "P2");
    await user.click(screen.getByRole("button", { name: "Why this?" }));
    await user.click(screen.getByRole("button", { name: "Mark out of stock" }));
    expect(noop.onOutOfStock).toHaveBeenCalledWith("P1");
    await user.click(screen.getByRole("button", { name: "Remove item" }));
    expect(noop.onRemove).toHaveBeenCalledWith("living_room.sofa");
  });

  it("disables destructive actions while busy", async () => {
    const user = userEvent.setup();
    render(<ItemCard item={item()} shortlist={[]} cartIds={[]} busy {...noop} />);
    await user.click(screen.getByRole("button", { name: "Why this?" }));
    expect(screen.getByRole("button", { name: "Mark out of stock" })).toBeDisabled();
  });

  it("shows the score as an accessible meter", () => {
    render(<ItemCard item={item()} shortlist={[]} cartIds={[]} busy={false} {...noop} />);
    expect(screen.getByRole("meter", { name: /Utility score/ })).toHaveAttribute("aria-valuenow", "59");
  });

  it("survives a broken image", () => {
    render(<ItemCard item={item({ product: product({ image_url: "https://x.example/a.jpg" }) })} shortlist={[]} cartIds={[]} busy={false} {...noop} />);
    screen.getByRole("img", { name: "Oak Sofa" }).dispatchEvent(new Event("error"));
    return waitFor(() => expect(screen.queryByRole("img", { name: "Oak Sofa" })).not.toBeInTheDocument());
  });
});

describe("ChatPanel", () => {
  it("sends trimmed text and clears the box", async () => {
    const user = userEvent.setup();
    const onSend = vi.fn();
    render(<ChatPanel history={[]} hasPlan={false} busy={false} onSend={onSend} />);
    await user.type(screen.getByLabelText("Message SmartNest"), "  2BHK 2 lakh  {enter}");
    expect(onSend).toHaveBeenCalledWith("2BHK 2 lakh");
    expect(screen.getByLabelText("Message SmartNest")).toHaveValue("");
  });

  it("ignores blank input and is locked while busy", async () => {
    const user = userEvent.setup();
    const onSend = vi.fn();
    const { rerender } = render(<ChatPanel history={[]} hasPlan={false} busy={false} onSend={onSend} />);
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
    await user.type(screen.getByLabelText("Message SmartNest"), "   {enter}");
    expect(onSend).not.toHaveBeenCalled();
    rerender(<ChatPanel history={[]} hasPlan={false} busy onSend={onSend} />);
    expect(screen.getByRole("status")).toHaveTextContent("thinking");
  });

  it("offers the example before a plan and follow-ups after", async () => {
    const user = userEvent.setup();
    const onSend = vi.fn();
    const { rerender } = render(<ChatPanel history={[]} hasPlan={false} busy={false} onSend={onSend} />);
    await user.click(screen.getByRole("button", { name: "Try the 2BHK example" }));
    expect(onSend.mock.calls[0][0]).toContain("2BHK");
    rerender(<ChatPanel history={[]} hasPlan busy={false} onSend={onSend} />);
    expect(screen.getByRole("button", { name: "Why this sofa?" })).toBeInTheDocument();
  });

  it("renders both roles and error replies", () => {
    render(<ChatPanel hasPlan busy={false} onSend={vi.fn()} history={[
      { role: "user", text: "hello", kind: "", time: "" }, { role: "assistant", text: "nope", kind: "error", time: "" }]} />);
    expect(screen.getByText("hello")).toBeInTheDocument();
    expect(screen.getByText("nope")).toHaveClass("text-bad");
  });
});

describe("BudgetView", () => {
  const budget: Budget = {
    total: 200000, spent: 100000, remaining: 100000, utilisation: 0.5, within_budget: true,
    by_room: [{ name: "Living room", value: 100000 }], by_category: [{ name: "Sofa", value: 100000 }],
    items: [{ label: "Living room sofa", room: "Living room", category: "Sofa", ceiling: 32000, price: 18628, status: "unverified" }],
    allocation_vs_spend: [{ room: "Living room", allocated: 32000, spent: 18628 }],
    counts: { confirmed: 0, unverified: 1, unfilled: 0 },
  };

  it("shows the headline numbers and the item table", async () => {
    render(<BudgetView budget={budget} />);
    expect(screen.getByText("₹2,00,000")).toBeInTheDocument();
    expect(screen.getAllByText("₹1,00,000").length).toBe(2);
    expect(screen.getByRole("meter", { name: "Budget used" })).toHaveAttribute("aria-valuenow", "50");
    expect(screen.getByRole("cell", { name: "Living room sofa" })).toBeInTheDocument();
  });

  it("says so when there is no plan", () => {
    render(<BudgetView budget={null} />);
    expect(screen.getByText("No budget yet")).toBeInTheDocument();
  });
});

describe("api client", () => {
  const respond = (status: number, body: unknown) =>
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }));

  it("returns parsed JSON", async () => {
    respond(200, { status: "ok" });
    expect((await api.health()).status).toBe("ok");
  });

  it("surfaces the server's message", async () => {
    respond(422, { detail: "that product cannot be bought" });
    await expect(api.event("s", { type: "remove_item", slot_id: "x" })).rejects.toMatchObject({ status: 422, message: "that product cannot be bought" });
  });

  it("reads FastAPI validation errors", async () => {
    respond(422, { detail: [{ msg: "Input should be greater than 0" }] });
    await expect(api.createSession({ total_budget: -1 })).rejects.toThrow("Input should be greater than 0");
  });

  it("gives a generic message for non-JSON failures", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("<html>", { status: 502 }));
    await expect(api.health()).rejects.toThrow("Request failed (502)");
  });

  it("reports an unreachable server", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("fetch failed"));
    const err = await api.health().catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.message).toMatch(/Cannot reach/);
  });

  it("encodes ids in paths", async () => {
    const spy = respond(200, {});
    await api.getSession("a/b c");
    expect(spy.mock.calls[0][0]).toBe("/api/sessions/a%2Fb%20c");
  });
});

import { CartView, cartText } from "./components/CartView";
import type { Cart, CartLine } from "./types";

const line = (over: Partial<CartLine> = {}, prod: Partial<Product> = {}): CartLine => ({
  product: product(prod), qty: 2, line_total: 37256, category: "sofa", status: "unverified", in_plan: true, out_of_stock: false, ...over,
});

const cartOf = (over: Partial<Cart> = {}): Cart => ({
  lines: [line()], count: 2, total: 37256, budget: 200000, remaining: 162744, within_budget: true, unverified: 1, warnings: [], ...over,
});

const cartActions = { onSetQty: vi.fn(), onRemove: vi.fn(), onClear: vi.fn(), onAddPlan: vi.fn() };

describe("ItemCard cart button", () => {
  it("adds to the cart and shows when it is already there", async () => {
    const user = userEvent.setup();
    const { rerender } = render(<ItemCard item={item()} shortlist={[]} cartIds={[]} busy={false} {...noop} />);
    await user.click(screen.getByRole("button", { name: "Add to cart" }));
    expect(noop.onAddToCart).toHaveBeenCalledWith("P1");
    rerender(<ItemCard item={item()} shortlist={[]} cartIds={["P1"]} busy={false} {...noop} />);
    expect(screen.getByRole("button", { name: "In cart" })).toBeDisabled();
  });

  it("can add an alternative to the cart", async () => {
    const user = userEvent.setup();
    const alt = { ...product({ product_id: "P2", name: "Pine Sofa" }), state: "unverified" as const, utility: 50, completeness: 0.4, gaps: [] };
    render(<ItemCard item={item({ alternatives: [alt] })} shortlist={[]} cartIds={[]} busy={false} {...noop} />);
    await user.click(screen.getByRole("button", { name: /Alternatives/ }));
    await user.click(screen.getByRole("button", { name: "Add Pine Sofa to cart" }));
    expect(noop.onAddToCart).toHaveBeenCalledWith("P2");
  });
});

describe("CartView", () => {
  it("shows lines, totals and honest stock labels", () => {
    render(<CartView cart={cartOf()} busy={false} {...cartActions} />);
    expect(screen.getAllByText("₹37,256").length).toBeGreaterThan(0);
    expect(screen.queryByText("Stock unverified")).not.toBeInTheDocument();
    expect(screen.queryByText("Confirmed in stock")).not.toBeInTheDocument();
    expect(screen.getByRole("meter", { name: "Cart against budget" })).toHaveAttribute("aria-valuenow", "19");
    expect(screen.getByText(/does not sell anything/)).toBeInTheDocument();
  });

  it("changes quantity, removes and clears", async () => {
    const user = userEvent.setup();
    render(<CartView cart={cartOf()} busy={false} {...cartActions} />);
    await user.click(screen.getByRole("button", { name: "Increase quantity" }));
    expect(cartActions.onSetQty).toHaveBeenCalledWith("P1", 3);
    await user.click(screen.getByRole("button", { name: "Decrease quantity" }));
    expect(cartActions.onSetQty).toHaveBeenCalledWith("P1", 1);
    await user.click(screen.getByRole("button", { name: /Remove Oak Sofa from cart/ }));
    expect(cartActions.onRemove).toHaveBeenCalledWith("P1");
    await user.click(screen.getByRole("button", { name: "Clear cart" }));
    expect(cartActions.onClear).toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: "Add all plan picks" }));
    expect(cartActions.onAddPlan).toHaveBeenCalled();
  });

  it("stops the quantity at ten", () => {
    render(<CartView cart={cartOf({ lines: [line({ qty: 10 })] })} busy={false} {...cartActions} />);
    expect(screen.getByRole("button", { name: "Increase quantity" })).toBeDisabled();
  });

  it("warns when over budget or out of stock", () => {
    render(<CartView cart={cartOf({ within_budget: false, remaining: -5, warnings: ["The cart is over the current budget."],
      lines: [line({ out_of_stock: true })] })} busy={false} {...cartActions} />);
    expect(screen.getByRole("alert")).toHaveTextContent("over the current budget");
    expect(screen.getByText("Now out of stock")).toBeInTheDocument();
  });

  it("gives simulated lines a sample badge and no link", () => {
    render(<CartView cart={cartOf({ lines: [line({}, { simulated: true, buy_url: null, link_label: "Sample listing - not a real product" })] })}
      busy={false} {...cartActions} />);
    expect(screen.getByText("Sample data")).toBeInTheDocument();
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
  });

  it("links real lines safely", () => {
    render(<CartView cart={cartOf()} busy={false} {...cartActions} />);
    const link = screen.getByRole("link", { name: /Retailer page/ });
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
  });

  it("explains an empty cart and disables clearing it", () => {
    render(<CartView cart={cartOf({ lines: [], count: 0, total: 0, remaining: 200000, unverified: 0 })} busy={false} {...cartActions} />);
    expect(screen.getByText("Your cart is empty")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Clear cart" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Copy buy list" })).toBeDisabled();
  });

  it("handles no plan", () => {
    render(<CartView cart={null} busy={false} {...cartActions} />);
    expect(screen.getByText("No cart yet")).toBeInTheDocument();
  });

  it("writes a plain-text buy list without links for sample data", () => {
    const text = cartText(cartOf({ lines: [line(), line({}, { product_id: "S", name: "Sample", simulated: true, buy_url: null })] }));
    expect(text).toContain("2 x Oak Sofa - ₹37,256 - https://www.wakefit.co/sofa");
    expect(text).toContain("(sample, no link)");
    expect(text).toContain("Total: ₹37,256 of ₹2,00,000 budget");
  });
});


describe("items without listings and themes", () => {
  const noData = () => item({ product: null, status: "unfilled", no_data: true, category: "study_table", category_label: "Study table",
    search_query: "study table", why: "SmartNest has no study table listings yet." });

  it("offers a live search that names its cost and query", async () => {
    const user = userEvent.setup();
    render(<ItemCard item={noData()} shortlist={[]} cartIds={[]} busy={false} live={{ enabled: true, remaining: 4, limit: 10, used: 6 }} {...noop} />);
    expect(screen.getByText("No listings yet")).toBeInTheDocument();
    expect(screen.getByText(/4 SerpApi request\(s\) left/)).toBeInTheDocument();
    expect(screen.getByText(/study table/, { selector: "span" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Search live for Study table" }));
    expect(noop.onSearch).toHaveBeenCalledWith("study_table");
  });

  it("disables the search when the allowance is used up and says why", () => {
    render(<ItemCard item={noData()} shortlist={[]} cartIds={[]} busy={false} live={{ enabled: true, remaining: 0, limit: 10, used: 10 }} {...noop} />);
    expect(screen.getByRole("button", { name: /Search live/ })).toBeDisabled();
    expect(screen.getByText(/allowance is used up \(10 of 10\)/)).toBeInTheDocument();
  });

  it("disables the search when live search is switched off", () => {
    render(<ItemCard item={noData()} shortlist={[]} cartIds={[]} busy={false} live={{ enabled: false }} {...noop} />);
    expect(screen.getByRole("button", { name: /Search live/ })).toBeDisabled();
    expect(screen.getByText(/switched off/)).toBeInTheDocument();
  });

  it("shows which theme words a listing's title mentions, and nothing when none", () => {
    const { rerender } = render(<ItemCard item={item({ product: product({ theme_matches: ["monster"] }) })} shortlist={[]} cartIds={[]} busy={false} {...noop} />);
    expect(screen.getByText("Mentions monster")).toBeInTheDocument();
    rerender(<ItemCard item={item({ product: product({ theme_matches: [] }) })} shortlist={[]} cartIds={[]} busy={false} {...noop} />);
    expect(screen.queryByText(/Mentions/)).not.toBeInTheDocument();
  });

  it("a genuinely unfilled item (not missing data) has no search button", () => {
    render(<ItemCard item={item({ product: null, status: "unfilled", why: "No sofa fits." })} shortlist={[]} cartIds={[]} busy={false} {...noop} />);
    expect(screen.queryByRole("button", { name: /Search live/ })).not.toBeInTheDocument();
    expect(screen.getByText("Not filled")).toBeInTheDocument();
  });
});

describe("an item that does not fit the budget", () => {
  const tight = () => item({ product: null, status: "unfilled", category: "wardrobe", category_label: "Wardrobe", cheapest_price: 12319,
    why: "The cheapest wardrobe listing costs ₹12,319, but only ₹7,249 of your budget is left." });

  it("explains itself and offers the two ways out", async () => {
    const user = userEvent.setup();
    const onRaise = vi.fn();
    render(<ItemCard item={tight()} shortlist={[]} cartIds={[]} busy={false} raiseTo={30000} onRaiseBudget={onRaise} {...noop} />);
    expect(screen.getByText(/cheapest wardrobe listing costs ₹12,319/)).toBeInTheDocument();
    expect(screen.getByText("Cheapest listing: ₹12,319")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Raise budget to ₹30,000" }));
    expect(onRaise).toHaveBeenCalledWith(30000);
    await user.click(screen.getByRole("button", { name: "Remove Wardrobe from the plan" }));
    expect(noop.onRemove).toHaveBeenCalledWith("living_room.sofa");
  });

  it("offers no budget button when raising the budget would not help", () => {
    render(<ItemCard item={tight()} shortlist={[]} cartIds={[]} busy={false} raiseTo={null} onRaiseBudget={vi.fn()} {...noop} />);
    expect(screen.queryByRole("button", { name: /Raise budget/ })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Remove Wardrobe/ })).toBeInTheDocument();
  });

  it("locks both actions while busy", () => {
    render(<ItemCard item={tight()} shortlist={[]} cartIds={[]} busy raiseTo={30000} onRaiseBudget={vi.fn()} {...noop} />);
    expect(screen.getByRole("button", { name: /Raise budget/ })).toBeDisabled();
    expect(screen.getByRole("button", { name: /Remove Wardrobe/ })).toBeDisabled();
  });
});


import { PlanView } from "./components/PlanView";
import type { RoomView } from "./types";

describe("themes and live refresh controls", () => {
  const live = { enabled: true, remaining: null, limit: null, used: 19 };
  const plain = () => item({ category: "rug", category_label: "Rug", theme_unmatched: true, search_query: "princess barbie kids rug" });

  it("says a generic pick is not themed and offers the themed search", async () => {
    const user = userEvent.setup();
    render(<ItemCard item={plain()} shortlist={[]} cartIds={[]} busy={false} live={live} {...noop} />);
    expect(screen.getByTestId("theme-note")).toHaveTextContent("Not themed");
    await user.click(screen.getByRole("button", { name: /Search live for “princess barbie kids rug” \(1 request\)/ }));
    expect(noop.onSearch).toHaveBeenCalledWith("rug");
  });

  it("only describes the themed search when live search is off", () => {
    render(<ItemCard item={plain()} shortlist={[]} cartIds={[]} busy={false} live={{ enabled: false }} {...noop} />);
    expect(screen.getByTestId("theme-note")).toHaveTextContent("A themed search would be “princess barbie kids rug”");
    expect(screen.queryByRole("button", { name: /Search live/ })).not.toBeInTheDocument();
  });

  it("shows no note for a themed or unthemed request", () => {
    render(<ItemCard item={item({ theme_unmatched: false })} shortlist={[]} cartIds={[]} busy={false} live={live} {...noop} />);
    expect(screen.queryByTestId("theme-note")).not.toBeInTheDocument();
  });

  it("lets you refresh any item's listings live, only when live search is on", async () => {
    const user = userEvent.setup();
    const { rerender } = render(<ItemCard item={item()} shortlist={[]} cartIds={[]} busy={false} live={live} {...noop} />);
    await user.click(screen.getByRole("button", { name: "Why this?" }));
    await user.click(screen.getByRole("button", { name: "Refresh Sofa listings" }));
    expect(noop.onSearch).toHaveBeenCalledWith("sofa");
    rerender(<ItemCard item={item()} shortlist={[]} cartIds={[]} busy={false} live={{ enabled: false }} {...noop} />);
    expect(screen.queryByRole("button", { name: /Refresh/ })).not.toBeInTheDocument();
  });

  const room = (items: ReturnType<typeof item>[]): RoomView => ({ room_id: "r", label: "Kid's bedroom", spent: 0, allocated: 0, filled: 0, count: items.length, items });
  const needy = [
    item({ slot_id: "r.wall_decor", category: "wall_decor", category_label: "Wall decor", product: null, status: "unfilled", no_data: true }),
    item({ slot_id: "r.rug", category: "rug", category_label: "Rug", theme_unmatched: true }),
    item({ slot_id: "r.bed", category: "bed", category_label: "Bed" }),
  ];
  const props = { shortlist: [], cartIds: [], busy: false, notes: [], assumptions: [], ...noop };

  it("offers one button for everything that needs a live search and passes the categories", async () => {
    const user = userEvent.setup();
    const many = vi.fn();
    render(<PlanView rooms={[room(needy)]} live={live} onSearchMany={many} {...props} />);
    expect(screen.getByTestId("search-many")).toHaveTextContent("2 items need a live search");
    await user.click(screen.getByRole("button", { name: "Search live for all 2 (2 requests)" }));
    expect(many).toHaveBeenCalledWith(["wall_decor", "rug"]);
  });

  it("has no banner for a single item, when nothing needs it, or when live search is off", () => {
    const many = vi.fn();
    const { rerender } = render(<PlanView rooms={[room(needy.slice(0, 1))]} live={live} onSearchMany={many} {...props} />);
    expect(screen.queryByTestId("search-many")).not.toBeInTheDocument();
    rerender(<PlanView rooms={[room(needy.slice(2))]} live={live} onSearchMany={many} {...props} />);
    expect(screen.queryByTestId("search-many")).not.toBeInTheDocument();
    rerender(<PlanView rooms={[room(needy)]} live={{ enabled: false }} onSearchMany={many} {...props} />);
    expect(screen.queryByTestId("search-many")).not.toBeInTheDocument();
  });

  it("counts a category once even if several items need it", () => {
    const twice = [...needy, item({ slot_id: "r2.rug", category: "rug", category_label: "Rug", theme_unmatched: true })];
    render(<PlanView rooms={[room(twice)]} live={live} onSearchMany={vi.fn()} {...props} />);
    expect(screen.getByTestId("search-many")).toHaveTextContent("2 items need");
  });

  it("disables the banner while busy", () => {
    render(<PlanView rooms={[room(needy)]} live={live} onSearchMany={vi.fn()} {...props} busy />);
    expect(screen.getByRole("button", { name: /Search live for all/ })).toBeDisabled();
  });
});
