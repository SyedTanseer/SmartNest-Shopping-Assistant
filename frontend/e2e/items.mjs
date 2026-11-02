// A request for THINGS, not a room - including something SmartNest has never heard of (node e2e/items.mjs [baseUrl] [outDir]).
// Needs the real Gemini key: Gemini returns only category NAMES; SerpApi data (or "no listings yet") supplies the rest.
import { chromium } from "playwright";
import { mkdirSync } from "node:fs";

const base = process.argv[2] ?? "http://localhost:5173";
const out = process.argv[3] ?? "e2e/shots";
mkdirSync(out, { recursive: true });
const results = [];
const check = (name, ok, detail = "") => { results.push(ok); console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? " - " + detail : ""}`); };

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 950 } });
const problems = [];
page.on("console", (m) => { if (m.type() === "error") problems.push(m.text()); });
page.on("pageerror", (e) => problems.push(String(e)));

await page.goto(base);
await page.getByRole("log").waitFor();
await page.getByLabel("Message SmartNest").fill("I want to buy a telescope and a bed, my budget is 40000");
await page.getByRole("button", { name: "Send" }).click();
await page.locator("[data-testid^=item-]").first().waitFor({ timeout: 90000 });
const ids = await page.locator("[data-testid^=item-]").evaluateAll((els) => els.map((e) => e.getAttribute("data-testid")));
check("a plan for the two things asked for", ids.length === 2, ids.join(", "));
check("it is a shopping list, not a home", (await page.getByRole("heading", { level: 3 }).allInnerTexts()).some((t) => /Shopping list/.test(t)) && (await page.getByText("Living room", { exact: true }).count()) === 0);
const telescope = page.getByTestId("item-shopping_list.x_telescope");
check("the unknown thing became its own item", await telescope.isVisible());
check("with no data it says so instead of inventing a product", /No listings yet/.test(await telescope.innerText()) && (await telescope.locator(".text-lg").count()) === 0);
check("it names the search it would run", /telescope/.test(await telescope.innerText()));
const health = await (await page.request.get(`${base}/api/health`)).json();
const searchBtn = telescope.getByRole("button", { name: /Search live/ });
if (health.live_search?.enabled) {
  // Live search is on: the button works, asks before spending, and cancelling spends nothing.
  const usedBefore = health.serpapi_quota?.used;
  let asked = "";
  page.once("dialog", async (d) => { asked = d.message(); await d.dismiss(); });
  check("live search is on: the button is enabled", await searchBtn.isEnabled());
  await searchBtn.click();
  await page.waitForTimeout(500);
  check("it asks before spending a request", /1 SerpApi request/.test(asked), asked);
  const after = await (await page.request.get(`${base}/api/health`)).json();
  check("cancelling spends nothing", after.serpapi_quota?.used === usedBefore && (await telescope.innerText()).includes("No listings yet"));
} else {
  check("the search is offered but guarded", await searchBtn.isDisabled());
}
const bed = page.getByTestId("item-shopping_list.bed");
check("the known thing uses real data", (await bed.locator(".text-lg").count()) === 1 && /Real listing/.test(await bed.innerText()));
const chat = await page.getByRole("log").innerText();
check("the chat explains the missing data", /no product listings yet|telescope/i.test(chat));
await page.screenshot({ path: `${out}/09-items.png`, fullPage: true });

await page.reload();
await page.getByTestId("item-shopping_list.x_telescope").waitFor({ timeout: 30000 });
check("the custom item survives a reload", (await page.locator("[data-testid^=item-]").count()) === 2);

await page.getByLabel("Message SmartNest").fill("I don't need a telescope");
await page.getByRole("button", { name: "Send" }).click();
await page.getByTestId("item-shopping_list.x_telescope").waitFor({ state: "detached", timeout: 60000 }).catch(() => {});
check("and it can be removed by name", (await page.locator("[data-testid^=item-]").count()) <= 2);

check("no browser console errors", problems.length === 0, problems.slice(0, 3).join(" | "));
await browser.close();
const failed = results.filter((r) => !r).length;
console.log(`\n${results.length - failed}/${results.length} checks passed`);
process.exit(failed ? 1 : 0);
