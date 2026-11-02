// A request that is NOT a whole home: one child's bedroom with themes (node e2e/kids-room.mjs [baseUrl] [outDir]).
import { chromium } from "playwright";
import { mkdirSync } from "node:fs";

const base = process.argv[2] ?? "http://localhost:5173";
const out = process.argv[3] ?? "e2e/shots";
mkdirSync(out, { recursive: true });
const PROMPT = "i want to decorate my 10 year old son's bedroom  under 50000, please find things that would be good for his bedroom, he is very much interested in cartoons and monsters as such so that would be a nice addition";
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
await say(PROMPT);
await page.locator("[data-testid^=item-]").first().waitFor({ timeout: 60000 });
const cards = await page.locator("[data-testid^=item-]").count();
check("one room, eleven items (not a whole home)", cards === 11, `${cards} items`);
const headings = (await page.getByRole("heading", { level: 3 }).allInnerTexts()).filter((t) => /room|Kitchen|Study|Bedroom/i.test(t));
check("only the kid's bedroom is planned", headings.length === 1 && /Kid's bedroom/.test(headings[0]), headings.join(", "));
check("no living room or kitchen invented", (await page.getByText("Living room", { exact: true }).count()) === 0 && (await page.getByText("Kitchen", { exact: true }).count()) === 0);
const reply = await page.getByRole("log").innerText();
check("the assistant echoes age, budget and the plan", /10-year-old/.test(reply) && /₹50,000/.test(reply));
check("every item is filled from real listings", (await page.locator("[data-testid^=item-] .text-lg").count()) === 11 && (await page.getByText("No listings yet").count()) === 0);
check("no sample data in a real-data plan", (await page.getByText("Sample data", { exact: true }).count()) === 0);
const themed = await page.getByText(/^Mentions /).allInnerTexts();
check("themed picks say what their title mentions", themed.length >= 2 && themed.some((t) => /monster/.test(t)) && themed.some((t) => /cartoon/.test(t)), themed.join(" | "));
await page.getByRole("tab", { name: "Budget" }).click();
await page.getByText("Planned spend").waitFor();
const spent = Number((await page.getByText("Planned spend").locator("xpath=following-sibling::div").first().innerText()).replace(/[^0-9]/g, ""));
check("the plan stays inside the budget", spent > 0 && spent <= 50000, `spent ${spent}`);
await page.getByRole("tab", { name: "Plan" }).click();
await page.screenshot({ path: `${out}/07-kids-room.png`, fullPage: true });

// A follow-up that used to build a 1BHK now just works inside the room.
await say("reduce the budget to 30000");
check("changes still work in a room plan", (await page.getByText(/₹30,000/).count()) > 0);
await say("show my cart");
await say("add the mattress to my cart");
await page.getByRole("tab", { name: /Cart/ }).click();
await page.getByText("Cart total").waitFor();
check("cart works for a room plan", (await page.locator("[data-testid^=cart-]").count()) === 1);

// A vague first message is answered with a question, then a bare number completes it.
await page.getByRole("button", { name: "New plan" }).click();
await page.waitForTimeout(800);
await page.getByRole("tab", { name: "Plan" }).click();
await say("decorate my daughter's room, she loves unicorns");
check("asks for the budget instead of guessing", /budget/i.test(await page.getByRole("log").innerText()) && (await page.locator("[data-testid^=item-]").count()) === 0);
await say("40000");
await page.locator("[data-testid^=item-]").first().waitFor({ timeout: 60000 });
check("a bare number answers the question", (await page.locator("[data-testid^=item-]").count()) === 11);

check("no browser console errors", problems.length === 0, problems.slice(0, 3).join(" | "));
await browser.close();
const failed = results.filter((r) => !r).length;
console.log(`\n${results.length - failed}/${results.length} checks passed`);
process.exit(failed ? 1 : 0);
