// Dark mode end to end (node e2e/dark-mode.mjs [baseUrl] [outDir]):
//   - follows the system setting on a first visit, the toggle switches and the choice survives a reload
//   - every tab is readable in both themes: a WCAG contrast audit over all visible text (including chart labels)
import { chromium } from "playwright";
import { mkdirSync } from "node:fs";

const base = process.argv[2] ?? "http://localhost:5173";
const out = process.argv[3] ?? "e2e/shots";
mkdirSync(out, { recursive: true });
const DEMO = "I want to decorate my 10 year old son's bedroom under 50000, he loves cartoons and monsters";
const results = [];
const check = (name, ok, detail = "") => { results.push(ok); console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? " - " + detail : ""}`); };

/** Runs in the page: contrast of every visible text element against its effective background. */
function audit() {
  const parse = (c) => {
    let m = c.match(/rgba?\(([^)]+)\)/);
    if (m) { const p = m[1].split(/[ ,/]+/).filter(Boolean).map(Number); return [p[0], p[1], p[2], p[3] ?? 1]; }
    m = c.match(/color\(srgb ([^)]+)\)/);
    if (m) { const p = m[1].split(/[ /]+/).filter(Boolean).map(Number); return [p[0] * 255, p[1] * 255, p[2] * 255, p[3] ?? 1]; }
    return null;
  };
  const over = (top, bottom) => {
    const a = top[3];
    return [top[0] * a + bottom[0] * (1 - a), top[1] * a + bottom[1] * (1 - a), top[2] * a + bottom[2] * (1 - a), 1];
  };
  const lum = ([r, g, b]) => {
    const f = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; };
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b);
  };
  const bgOf = (el) => {
    const layers = [];
    for (let e = el; e; e = e.parentElement) {
      const c = parse(getComputedStyle(e).backgroundColor);
      if (c && c[3] > 0) { layers.push(c); if (c[3] >= 1) break; }
    }
    let base = parse(getComputedStyle(document.body).backgroundColor) ?? [255, 255, 255, 1];
    if (base[3] < 1) base = [255, 255, 255, 1];
    return layers.reduceRight((acc, l) => over(l, acc), base);
  };
  const bad = [];
  let checked = 0;
  const all = document.querySelectorAll("body *");
  for (const el of all) {
    const own = [...el.childNodes].some((n) => n.nodeType === 3 && n.textContent.trim());
    if (!own) continue;
    const st = getComputedStyle(el);
    if (st.visibility === "hidden" || st.display === "none" || el.closest("[disabled]") || el.closest("[aria-hidden=true]:not(svg *)")) continue;
    const rect = el.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) continue;
    const isSvg = el instanceof SVGElement;
    const fg = parse(isSvg ? st.fill : st.color);
    if (!fg) continue;
    const bg = bgOf(el);
    const eff = fg[3] < 1 ? over(fg, bg) : fg;
    const ratio = (Math.max(lum(eff), lum(bg)) + 0.05) / (Math.min(lum(eff), lum(bg)) + 0.05);
    const size = parseFloat(st.fontSize), bold = parseInt(st.fontWeight, 10) >= 700;
    const need = size >= 24 || (size >= 18.66 && bold) ? 3 : 4.5;
    checked++;
    if (ratio < need) bad.push(`${ratio.toFixed(2)} < ${need} "${el.textContent.trim().slice(0, 40)}" (${isSvg ? "svg " : ""}${el.tagName.toLowerCase()})`);
  }
  return { checked, bad };
}

const browser = await chromium.launch();
const problems = [];

async function session(scheme) {
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 950 }, colorScheme: scheme });
  const page = await ctx.newPage();
  page.on("console", (m) => { if (m.type() === "error") problems.push(m.text()); });
  page.on("pageerror", (e) => problems.push(String(e)));
  return { ctx, page };
}
const theme = (page) => page.evaluate(() => document.documentElement.getAttribute("data-theme"));
const bodyBg = (page) => page.evaluate(() => getComputedStyle(document.body).backgroundColor);

// 1. First visit follows the system setting
{
  const dark = await session("dark");
  await dark.page.goto(base);
  await dark.page.getByRole("log").waitFor();
  await dark.page.waitForTimeout(400);
  check("first visit follows a dark system setting", (await theme(dark.page)) === "dark" && (await bodyBg(dark.page)) === "rgb(18, 17, 15)", await bodyBg(dark.page));
  const light = await session("light");
  await light.page.goto(base);
  await light.page.getByRole("log").waitFor();
  await light.page.waitForTimeout(400);
  check("first visit follows a light system setting", (await theme(light.page)) === "light" && (await bodyBg(light.page)) === "rgb(250, 248, 245)");
  await light.ctx.close();

  // 2. The toggle, and the choice survives a reload
  const toggle = dark.page.getByRole("button", { name: "Switch to light mode" });
  await toggle.click();
  await dark.page.getByRole("button", { name: "Switch to dark mode" }).waitFor();
  check("the toggle switches to light and relabels itself", (await theme(dark.page)) === "light");
  await dark.page.reload();
  await dark.page.getByRole("log").waitFor();
  check("the choice is remembered after a reload (overriding the system)", (await theme(dark.page)) === "light");
  await dark.page.getByRole("button", { name: "Switch to dark mode" }).click();
  check("and it switches back", (await theme(dark.page)) === "dark");
  await dark.ctx.close();
}

// 3. Contrast audit of every tab, both themes, on a real plan
for (const scheme of ["dark", "light"]) {
  const { ctx, page } = await session(scheme);
  await page.goto(base);
  await page.getByRole("log").waitFor();
  await page.getByLabel("Message SmartNest").fill(DEMO);
  await page.getByRole("button", { name: "Send" }).click();
  await page.getByTestId("item-kids_bedroom.wall_decor").waitFor({ timeout: 90000 });
  await page.waitForTimeout(1500);                                     // thumbnails
  const first = page.getByTestId("item-kids_bedroom.rug");
  await first.getByRole("button", { name: "Add to cart" }).click();
  await first.getByRole("button", { name: "Why this?" }).click();
  await page.getByTestId("item-kids_bedroom.mattress").getByRole("button", { name: /Alternatives/ }).click();
  await page.getByTestId("item-kids_bedroom.rug").getByRole("button", { name: "Add to shortlist" }).click();
  const tabs = [["Plan", "plan"], ["Budget", "budget"], ["Compare", "compare"], ["Cart", "cart"], ["Agents", "agents"]];
  for (const [name, file] of tabs) {
    await page.getByRole("tab", { name: new RegExp(name) }).click();
    if (name === "Budget") { await page.getByText("Spend by room").waitFor(); await page.waitForTimeout(1200); }
    if (name === "Compare") { for (const b of await page.getByRole("checkbox").all()) await b.check(); await page.waitForTimeout(500); }
    if (name === "Agents") { await page.getByText("Activity timeline").waitFor(); }
    await page.screenshot({ path: `${out}/dm-${scheme}-${file}.png`, fullPage: name !== "Plan" });
    const { checked, bad } = await page.evaluate(audit);
    check(`${scheme}: ${name} tab is readable (${checked} text elements)`, bad.length === 0, bad.slice(0, 4).join(" | "));
  }
  await page.getByRole("tab", { name: "Plan" }).click();
  await page.screenshot({ path: `${out}/dm-${scheme}-plan-full.png`, fullPage: true });
  await ctx.close();
}

check("no browser console errors", problems.length === 0, problems.slice(0, 3).join(" | "));
await browser.close();
const failed = results.filter((r) => !r).length;
console.log(`\n${results.length - failed}/${results.length} checks passed`);
process.exit(failed ? 1 : 0);
