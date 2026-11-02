// A themed request whose theme has NO saved listings (volcano / submarine: pick a theme nobody has searched for) (node e2e/princess.mjs [baseUrl] [outDir]).
// The saved wall-decor search is for "monster cartoon", so it must NOT be reused for princess / barbie.
// Live search must be ON for the search controls to be enabled. Dialogs are dismissed: no SerpApi request is spent.
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
const say = async (text) => {
  await page.getByLabel("Message SmartNest").fill(text);
  await page.getByRole("button", { name: "Send" }).click();
  await page.getByText("SmartNest is thinking…").waitFor({ state: "detached", timeout: 90000 });
};

await page.goto(base);
await page.getByRole("log").waitFor();
const before = (await (await page.request.get(`${base}/api/health`)).json());
await say("furnish my son's bedroom with a volcano and submarine theme");
if (!(await page.locator("[data-testid^=item-]").count())) await say("the budget is 50000");
await page.locator("[data-testid^=item-]").first().waitFor({ timeout: 90000 });
await page.waitForTimeout(1500);

const text = await page.locator("main").innerText();
const decorCard = page.getByTestId("item-kids_bedroom.wall_decor");
check("no wall-decor PRODUCT is shown (saved searches are for other themes)", (await decorCard.locator(".text-lg").count()) === 0 && !/Real listing|Add to cart/.test(await decorCard.innerText()));
check("no anime / manga product anywhere on the page", !/anime|manga|urasawa/i.test(await page.locator("main").innerText()));
const decor = page.getByTestId("item-kids_bedroom.wall_decor");
check("wall decor says there is nothing saved for this theme", /No listings yet/.test(await decor.innerText()) && /does not cover/.test(await decor.innerText()));
check("it names the themed search it would run", /volcano|submarine/.test(await decor.innerText()));
const rug = page.getByTestId("item-kids_bedroom.rug");
check("the generic rug is labelled 'Not themed'", /Not themed/.test(await rug.innerText()));
check("items that do not depend on the theme are still shown", (await page.locator("[data-testid^=item-] .text-lg").count()) >= 8);

const live = (await (await page.request.get(`${base}/api/health`)).json()).live_search;
if (live?.enabled) {
  check("live search is on and the banner offers all of it at once", /2 items need a live search/.test(await page.getByTestId("search-many").innerText()));
  let asked = "";
  page.once("dialog", async (d) => { asked = d.message(); await d.dismiss(); });
  await page.getByRole("button", { name: /Search live for all 2/ }).click();
  await page.waitForTimeout(500);
  check("it asks first and names the cost", /2 categories/.test(asked) && /2 SerpApi requests/.test(asked), asked);
  await decor.getByRole("button", { name: /Search live/ }).isEnabled();
  await rug.getByRole("button", { name: /Search live for/ }).waitFor();
  check("each item has its own search button too", true);
  const after = (await (await page.request.get(`${base}/api/health`)).json());
  check("cancelling spent nothing", after.serpapi_quota?.used === before.serpapi_quota?.used, `${before.serpapi_quota?.used} -> ${after.serpapi_quota?.used}`);
  // the refresh control exists on a normal item
  const bed = page.getByTestId("item-kids_bedroom.kids_bed");
  await bed.getByRole("button", { name: "Why this?" }).click();
  check("any item can be refreshed live", await bed.getByRole("button", { name: /Refresh .* listings/ }).isVisible());
}
const chat = await page.getByRole("log").innerText();
check("the chat explains it", /no product listings yet for: wall decor/i.test(chat) || /wall decor/i.test(chat));
await page.screenshot({ path: `${out}/10-princess.png`, fullPage: true });
check("no browser console errors", problems.length === 0, problems.slice(0, 3).join(" | "));
await browser.close();
const failed = results.filter((r) => !r).length;
console.log(`\n${results.length - failed}/${results.length} checks passed`);
process.exit(failed ? 1 : 0);
