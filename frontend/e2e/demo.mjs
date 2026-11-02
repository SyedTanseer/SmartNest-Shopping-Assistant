// End-to-end check of the real UI against the real API (node e2e/demo.mjs [baseUrl] [outDir]).
// Plays the final demo scenario, exercises every tab and screenshots each step. Exits 1 on any failed check.
import { chromium } from "playwright";
import { mkdirSync } from "node:fs";

const base = process.argv[2] ?? "http://localhost:5173";
const out = process.argv[3] ?? "e2e/shots";
mkdirSync(out, { recursive: true });

const DEMO = "I just moved into a 2BHK apartment. My furnishing budget is ₹2 lakh. I prefer minimalist furniture. Help me furnish the entire apartment.";
const results = [];
const check = (name, ok, detail = "") => { results.push(ok); console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? " - " + detail : ""}`); };

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 950 } });
const problems = [];
page.on("console", (m) => { if (m.type() === "error") problems.push(m.text()); });
page.on("pageerror", (e) => problems.push(String(e)));

await page.goto(base);
await page.getByRole("log").waitFor();
check("app loads with chat and status badges", (await page.getByLabel("System status").innerText()).includes("Real data"));
await page.screenshot({ path: `${out}/01-empty.png` });

await page.getByLabel("Message SmartNest").fill(DEMO);
await page.getByRole("button", { name: "Send" }).click();
await page.getByTestId("item-living_room.sofa").waitFor({ timeout: 60000 });
const cards = await page.locator("[data-testid^=item-]").count();
check("demo message builds a 10-item plan", cards === 10, `${cards} items`);
check("four rooms shown", (await page.getByRole("heading", { level: 3 }).allInnerTexts()).filter((t) => /Living room|Bedroom 1|Bedroom 2|Kitchen/.test(t)).length === 4);
const links = await page.locator("a[href^='https://']").count();
check("real listings link out safely", links > 0 && (await page.locator("a[target=_blank]:not([rel*=noopener])").count()) === 0, `${links} links`);
check("no 'Sample data' badge on real data", (await page.getByText("Sample data", { exact: true }).count()) === 0);
await page.screenshot({ path: `${out}/02-plan.png`, fullPage: true });

// Alternatives -> swap
const sofa = page.getByTestId("item-living_room.sofa");
const priceBefore = await sofa.locator(".text-lg").innerText();
await sofa.getByRole("button", { name: /Alternatives/ }).click();
const useBtn = sofa.getByRole("button", { name: /^Use / }).first();
await useBtn.click();
await sofa.getByText("Your choice").waitFor();
const priceAfter = await sofa.locator(".text-lg").innerText();
check("choosing an alternative replaces the pick", priceBefore !== priceAfter, `${priceBefore} -> ${priceAfter}`);

// Shortlist + compare
for (const id of ["living_room.sofa", "kitchen.refrigerator"]) {
  await page.getByTestId(`item-${id}`).getByRole("button", { name: "Add to shortlist" }).click();
  await page.getByTestId(`item-${id}`).getByRole("button", { name: "Remove from shortlist" }).waitFor();
}
await page.getByRole("tab", { name: /Compare/ }).click();
for (const box of await page.getByRole("checkbox").all()) await box.check();
await page.getByRole("columnheader", { name: /./ }).nth(2).waitFor();
check("compare table shows both products", (await page.getByRole("columnheader").count()) >= 3);
await page.screenshot({ path: `${out}/03-compare.png`, fullPage: true });

// Cart: add from a card, from the shortlist, then everything
await page.getByRole("tab", { name: "Plan" }).click();
await page.getByTestId("item-bedroom_1.bed").getByRole("button", { name: "Add to cart" }).click();
await page.getByTestId("item-bedroom_1.bed").getByRole("button", { name: "In cart" }).waitFor();
await page.getByRole("tab", { name: /Cart/ }).click();
await page.getByText("Cart total").waitFor();
check("adding to the cart shows one line", (await page.locator("[data-testid^=cart-]").count()) === 1);
await page.getByRole("button", { name: "Increase quantity" }).click();
await page.waitForFunction(() => document.querySelector("[aria-live=polite].w-8")?.textContent === "2");
check("quantity stepper works", true);
await page.getByRole("button", { name: "Add all plan picks" }).click();
await page.waitForFunction(() => document.querySelectorAll("[data-testid^=cart-]").length >= 7);
const total = await page.locator("text=Cart total").locator("xpath=following-sibling::div").first().innerText();
check("add-all fills the cart within budget", (await page.getByText("over the current budget").count()) === 0, `total ${total}`);
check("cart never claims confirmed without evidence", (await page.getByText("Confirmed in stock", { selector: "span" }).count()) === 0);
await page.screenshot({ path: `${out}/03b-cart.png`, fullPage: true });
await page.getByRole("button", { name: "Clear cart" }).click();
await page.getByText("Your cart is empty").first().waitFor();
check("clear cart empties it", (await page.locator("[data-testid^=cart-]").count()) === 0);
await page.getByLabel("Message SmartNest").fill("add the sofa to my cart");
await page.getByRole("button", { name: "Send" }).click();
await page.waitForFunction(() => document.querySelectorAll("[data-testid^=cart-]").length === 1, null, { timeout: 60000 });
check("chat can add to the cart", true);

// Budget
await page.getByRole("tab", { name: "Budget" }).click();
await page.getByText("Spend by room").waitFor();
await page.waitForTimeout(800);
check("budget charts render", (await page.locator(".recharts-surface").count()) >= 3);
check("budget table lists every item", (await page.locator("tbody tr").count()) === 10);
await page.screenshot({ path: `${out}/04-budget.png`, fullPage: true });

// Chat changes
await page.getByRole("tab", { name: "Plan" }).click();
await page.getByLabel("Message SmartNest").fill("I don't need a TV unit");
await page.getByRole("button", { name: "Send" }).click();
await page.getByTestId("item-living_room.tv_unit").waitFor({ state: "detached", timeout: 60000 });
check("chat command removes an item", (await page.locator("[data-testid^=item-]").count()) === 9);

await page.getByLabel("Message SmartNest").fill("Increase the budget to 3 lakh");
await page.getByRole("button", { name: "Send" }).click();
await page.getByText(/₹3,00,000/).first().waitFor({ timeout: 60000 });
check("chat command changes the budget", true);

// Out of stock via UI
const fridge = page.getByTestId("item-kitchen.refrigerator");
const fridgeBefore = await fridge.locator("h4").innerText();
await fridge.getByRole("button", { name: "Why this?" }).click();
await fridge.getByRole("button", { name: "Mark out of stock" }).click();
await page.waitForFunction(([sel, before]) => document.querySelector(sel)?.innerText !== before,
  ["[data-testid='item-kitchen.refrigerator'] h4", fridgeBefore]);
check("marking out of stock switches product", (await fridge.locator("h4").innerText()) !== fridgeBefore);

// Agents
await page.getByRole("tab", { name: "Agents" }).click();
await page.getByText("Activity timeline").waitFor();
check("agent roster shows five agents", (await page.getByRole("heading", { level: 3 }).filter({ hasText: /Coordinator|Space Planning|Product Research|Comparison|Budget/ }).count()) >= 5);
check("every message signature verified", (await page.getByText("✗ signature failed").count()) === 0 && (await page.getByText("✓ signature verified").count()) > 5);
await page.screenshot({ path: `${out}/05-agents.png`, fullPage: true });

// Reload keeps the session
await page.reload();
await page.getByRole("log").waitFor();
await page.getByTestId("item-kitchen.refrigerator").waitFor({ timeout: 30000 });
check("session survives a page reload", (await page.locator("[data-testid^=item-]").count()) === 9);

// Mobile layout: no horizontal scroll
await page.setViewportSize({ width: 390, height: 844 });
await page.waitForTimeout(300);
check("mobile layout has no horizontal scroll", await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1));
await page.screenshot({ path: `${out}/06-mobile.png` });

check("no browser console errors", problems.length === 0, problems.slice(0, 3).join(" | "));
await browser.close();
const failed = results.filter((r) => !r).length;
console.log(`\n${results.length - failed}/${results.length} checks passed`);
process.exit(failed ? 1 : 0);
