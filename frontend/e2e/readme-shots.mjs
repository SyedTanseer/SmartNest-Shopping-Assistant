// Takes the screenshots used in the root README (node e2e/readme-shots.mjs [baseUrl] [outDir]).
// Real stack, real Gemini, saved real listings. No live search is run, so nothing is spent.
import { chromium } from "playwright";
import { mkdirSync } from "node:fs";

const base = process.argv[2] ?? "http://localhost:5173";
const out = process.argv[3] ?? "../docs/screenshots";
mkdirSync(out, { recursive: true });
const browser = await chromium.launch();

async function open(theme) {
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, colorScheme: theme });
  await ctx.addInitScript((t) => { try { localStorage.setItem("smartnest.theme", t); } catch {} }, theme);
  const page = await ctx.newPage();
  await page.goto(base);
  await page.getByRole("log").waitFor();
  return { ctx, page };
}
async function say(page, text) {
  await page.getByLabel("Message SmartNest").fill(text);
  await page.getByRole("button", { name: "Send" }).click();
  await page.getByText("SmartNest is thinking…").waitFor({ state: "detached", timeout: 90000 });
}
const settle = (page, ms = 1800) => page.waitForTimeout(ms);          // thumbnails and chart animation

// 1. Dark: a themed request for a child's room (real saved Barbie / princess listings)
{
  const { ctx, page } = await open("dark");
  await say(page, "furnish my daughter's bedroom with princess and barbie theme");
  if (!(await page.locator("[data-testid^=item-]").count())) await say(page, "50000");
  await page.locator("[data-testid^=item-]").first().waitFor({ timeout: 90000 });
  await settle(page);
  await page.screenshot({ path: `${out}/plan-dark.png` });
  await page.getByTestId("item-kids_bedroom.wall_decor").scrollIntoViewIfNeeded();
  await page.mouse.wheel(0, -120);
  await settle(page, 800);
  await page.screenshot({ path: `${out}/themed-dark.png` });
  await page.getByRole("tab", { name: "Budget" }).click();
  await page.getByText("Spend by room").waitFor();
  await settle(page, 2200);
  await page.screenshot({ path: `${out}/budget-dark.png` });
  await page.getByRole("tab", { name: "Agents" }).click();
  await page.getByText("Activity timeline").waitFor();
  await settle(page, 800);
  await page.screenshot({ path: `${out}/agents-dark.png` });
  await ctx.close();
}

// 2. Light: a budget too small for everything (the honest explanation) and the cart
{
  const { ctx, page } = await open("light");
  await say(page, "decorate my 10 year old son's bedroom under 25000, he loves monsters and cartoons");
  await page.getByTestId("item-kids_bedroom.wardrobe").waitFor({ timeout: 90000 });
  await settle(page);
  await page.getByTestId("item-kids_bedroom.wardrobe").scrollIntoViewIfNeeded();
  await page.mouse.wheel(0, -160);
  await settle(page, 600);
  await page.screenshot({ path: `${out}/tight-budget-light.png` });
  await page.getByRole("tab", { name: /Cart/ }).click();
  await page.getByRole("button", { name: "Add all plan picks" }).click();
  await page.waitForFunction(() => document.querySelectorAll("[data-testid^=cart-]").length >= 5);
  await settle(page, 600);
  await page.screenshot({ path: `${out}/cart-light.png` });
  await ctx.close();
}
await browser.close();
console.log("saved to", out);
