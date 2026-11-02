// A budget too small for everything: the app must explain and offer a fix (node e2e/tight-budget.mjs [baseUrl] [outDir]).
import { chromium } from "playwright";
import { mkdirSync } from "node:fs";

const base = process.argv[2] ?? "http://localhost:5173";
const out = process.argv[3] ?? "e2e/shots";
mkdirSync(out, { recursive: true });
const PROMPT = "i want to decorate my 10 year old son's bedroom under a budget of 25000, he loves monsters and cartoons";
const results = [];
const check = (name, ok, detail = "") => { results.push(ok); console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? " - " + detail : ""}`); };

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 950 } });
const problems = [];
page.on("console", (m) => { if (m.type() === "error") problems.push(m.text()); });
page.on("pageerror", (e) => problems.push(String(e)));

await page.goto(base);
await page.getByRole("log").waitFor();
await page.getByLabel("Message SmartNest").fill(PROMPT);
await page.getByRole("button", { name: "Send" }).click();
await page.getByTestId("item-kids_bedroom.wardrobe").waitFor({ timeout: 90000 });
const wardrobe = page.getByTestId("item-kids_bedroom.wardrobe");
const text = await wardrobe.innerText();
check("the wardrobe is the one item left out", (await page.getByText("Not filled").count()) === 1 && /Wardrobe/.test(text));
check("the card says what the cheapest wardrobe costs and what is left", /₹12,319/.test(text) && /₹7,249/.test(text), text.replace(/\s+/g, " ").slice(0, 120));
const raise = wardrobe.getByRole("button", { name: /Raise budget to/ });
check("it offers a one-click fix", await raise.isVisible() && /₹30,000/.test(await raise.innerText()));
check("it offers to drop the item", await wardrobe.getByRole("button", { name: /Remove Wardrobe/ }).isVisible());
const chat = await page.getByRole("log").innerText();
check("the chat gives the numbers", /29,581/.test(chat) && /30,000/.test(chat) && /too small/.test(chat));
await page.screenshot({ path: `${out}/08-tight-budget.png`, fullPage: true });

await raise.click();
await page.getByTestId("item-kids_bedroom.wardrobe").locator("h4").waitFor({ timeout: 60000 });
await page.waitForFunction(() => document.querySelectorAll("[data-testid^=item-] .text-lg").length === 11, null, { timeout: 60000 });
check("after raising the budget all 11 items are filled", (await page.getByText("Not filled").count()) === 0);
await page.getByRole("tab", { name: "Budget" }).click();
await page.getByText("Planned spend").waitFor();
const spent = Number((await page.getByText("Planned spend").locator("xpath=following-sibling::div").first().innerText()).replace(/[^0-9]/g, ""));
check("and the spend is inside the new budget", spent > 0 && spent <= 30000, `spent ${spent}`);
check("no browser console errors", problems.length === 0, problems.slice(0, 3).join(" | "));
await browser.close();
const failed = results.filter((r) => !r).length;
console.log(`\n${results.length - failed}/${results.length} checks passed`);
process.exit(failed ? 1 : 0);
