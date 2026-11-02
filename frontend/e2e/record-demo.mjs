// Records the ~3 minute SmartNest demo: node e2e/record-demo.mjs [baseUrl] [outFile]
// A scripted, captioned run of the real UI against the real API (no mock data). Output is a silent .webm.
import { chromium } from "playwright";
import { mkdirSync, readdirSync, renameSync, rmSync } from "node:fs";
import { dirname } from "node:path";

const base = process.argv[2] ?? "http://localhost:5173";
const outFile = process.argv[3] ?? "../docs/demo/smartnest-demo.webm";
const tmp = "e2e/_video";
rmSync(tmp, { recursive: true, force: true });
mkdirSync(tmp, { recursive: true });
mkdirSync(dirname(outFile), { recursive: true });

const DEMO = "I just moved into a 2BHK apartment. My furnishing budget is ₹2 lakh. I prefer minimalist furniture. Help me furnish the entire apartment.";
const started = Date.now();
const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1366, height: 768 }, recordVideo: { dir: tmp, size: { width: 1366, height: 768 } } });
const page = await ctx.newPage();
const hold = (ms) => page.waitForTimeout(ms);

async function caption(text, ms) {
  await page.evaluate((t) => {
    let el = document.getElementById("demo-caption");
    if (!el) {
      el = document.createElement("div");
      el.id = "demo-caption";
      el.style.cssText = "position:fixed;left:50%;bottom:22px;transform:translateX(-50%);max-width:880px;z-index:99999;background:rgba(28,25,23,.92);color:#fff;padding:12px 20px;border-radius:12px;font:500 17px/1.4 Inter,system-ui,sans-serif;text-align:center;box-shadow:0 8px 24px rgba(0,0,0,.3)";
      document.body.appendChild(el);
    }
    el.textContent = t;
  }, text);
  await hold(ms);
}
const send = async (text, wait) => {
  await page.getByLabel("Message SmartNest").pressSequentially(text, { delay: 28 });
  await hold(500);
  await page.getByRole("button", { name: "Send" }).click();
  await page.getByText("SmartNest is thinking…").waitFor({ state: "detached", timeout: 90000 });
  await hold(wait);
};
const scrollTo = async (testid) => {
  await page.getByTestId(testid).scrollIntoViewIfNeeded();
  await hold(300);
};

await page.goto(base);
await page.getByRole("log").waitFor();
await caption("SmartNest: an autonomous planner that furnishes a whole home within your budget.", 5500);
await caption("Real Google Shopping listings, honest scores, and every agent decision on screen.", 3500);

await caption("1. Describe your home in plain language.", 2500);
await send(DEMO, 1500);
await page.getByTestId("item-living_room.sofa").waitFor({ timeout: 90000 });
await caption("Four specialist agents planned 10 items across 4 rooms, inside the ₹2 lakh budget.", 4500);

await caption("2. Room by room: real products, prices, scores and links.", 3500);
for (const id of ["living_room.sofa", "bedroom_1.bed", "bedroom_2.wardrobe", "kitchen.refrigerator"]) { await scrollTo(`item-${id}`); await hold(2200); }
await page.evaluate(() => window.scrollTo({ top: 0, behavior: "smooth" }));
await hold(1000);

await caption("Stock is only “confirmed” when the listing says so. Everything else is marked unverified.", 4500);
const sofa = page.getByTestId("item-living_room.sofa");
await sofa.getByRole("button", { name: "Why this?" }).click();
await caption("“Why this?” shows the score points and which attributes the listing did not state.", 5000);
await sofa.getByRole("button", { name: /Alternatives/ }).click();
await caption("Alternatives within the item's budget. Pick one and SmartNest re-checks the budget.", 4500);
await sofa.getByRole("button", { name: /^Use / }).first().click();
await sofa.getByText("Your choice").waitFor();
await caption("The plan updated; the total can never exceed the budget.", 4500);

await caption("3. Shortlist products and compare them side by side.", 3000);
for (const id of ["living_room.sofa", "kitchen.refrigerator"]) {
  await scrollTo(`item-${id}`);
  await page.getByTestId(`item-${id}`).getByRole("button", { name: "Add to shortlist" }).click();
  await hold(700);
}
await page.getByRole("tab", { name: /Compare/ }).click();
for (const box of await page.getByRole("checkbox").all()) { await box.check(); await hold(500); }
await hold(5000);

await caption("4. Add products to your cart. Anything over your budget is refused.", 3500);
await page.getByRole("tab", { name: "Plan" }).click();
await scrollTo("item-living_room.sofa");
await page.getByTestId("item-living_room.sofa").getByRole("button", { name: "Add to cart" }).click();
await hold(1200);
await page.getByRole("tab", { name: /Cart/ }).click();
await caption("The cart keeps a running total against your budget, with honest stock labels and links out to the seller.", 4500);
await page.getByRole("button", { name: "Add all plan picks" }).click();
await hold(1500);
await caption("One click adds every SmartNest pick. SmartNest sells nothing: each link opens the retailer to buy.", 5000);
await page.getByRole("button", { name: "Clear cart" }).click();
await hold(800);

await page.getByRole("tab", { name: "Budget" }).click();
await caption("5. The budget dashboard: where every rupee goes.", 4500);
await hold(3500);
await page.evaluate(() => window.scrollTo({ top: 520, behavior: "smooth" }));
await caption("Each item has a hard price ceiling. Unspent money stays yours.", 5000);
await page.evaluate(() => window.scrollTo({ top: 0, behavior: "smooth" }));

await page.getByRole("tab", { name: "Plan" }).click();
await caption("6. Change your mind in chat. SmartNest replans only what is affected.", 4500);
await send("I don't need a TV unit", 3500);
await caption("The TV unit is gone and the totals updated.", 3000);
await send("The fridge is out of stock", 3500);
await caption("An out-of-stock event: the Coordinator replaces the fridge and keeps everything else.", 5500);
await send("Reduce the budget to 1 lakh", 4000);
await caption("A budget cut: items are re-planned within ₹1 lakh. Nothing is forced; if it can't fit, it says so.", 6500);

await page.getByRole("tab", { name: "Agents" }).click();
await caption("7. The agent activity monitor.", 3500);
await hold(2500);
await caption("Every request between the agents is an Ed25519-signed message, verified before any agent acts.", 7000);
await page.evaluate(() => window.scrollTo({ top: 420, behavior: "smooth" }));
await caption("Replanning events from the Coordinator's beliefs, desires and intentions appear in the timeline.", 7000);

await caption("Real data · hard budget limits · honest uncertainty · signed agent messages.", 6000);
await caption("SmartNest: React · FastAPI · Gemini · MongoDB Atlas · SerpApi · custom Python agents.", 5000);

const seconds = Math.round((Date.now() - started) / 1000);
await ctx.close();
await browser.close();
const file = readdirSync(tmp).find((f) => f.endsWith(".webm"));
renameSync(`${tmp}/${file}`, outFile);
rmSync(tmp, { recursive: true, force: true });
console.log(`Saved ${outFile} (about ${seconds} s)`);
