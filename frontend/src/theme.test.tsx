import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const root = () => document.documentElement;

/** A fresh copy of the module, so its start-up logic (saved choice / system setting) can be tested. */
async function load(opts: { stored?: string | null; systemDark?: boolean; attr?: string | null; storageBroken?: boolean } = {}) {
  vi.resetModules();
  localStorage.clear();
  if (opts.stored) localStorage.setItem("smartnest.theme", opts.stored);
  if (opts.attr) root().setAttribute("data-theme", opts.attr); else root().removeAttribute("data-theme");
  vi.stubGlobal("matchMedia", (q: string) => ({
    matches: q.includes("dark") && !!opts.systemDark, addEventListener: vi.fn(), removeEventListener: vi.fn(),
  }));
  if (opts.storageBroken) {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => { throw new Error("blocked"); });
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => { throw new Error("blocked"); });
  }
  return import("./theme");
}

beforeEach(() => root().removeAttribute("data-theme"));
afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks(); localStorage.clear(); root().removeAttribute("data-theme"); });

describe("starting theme", () => {
  it("follows a dark system setting on a first visit", async () => {
    expect((await load({ systemDark: true })).initialTheme()).toBe("dark");
  });
  it("follows a light system setting on a first visit", async () => {
    expect((await load({ systemDark: false })).initialTheme()).toBe("light");
  });
  it("a saved choice beats the system setting", async () => {
    expect((await load({ stored: "light", systemDark: true })).initialTheme()).toBe("light");
    expect((await load({ stored: "dark", systemDark: false })).initialTheme()).toBe("dark");
  });
  it("trusts the attribute the page already set before first paint", async () => {
    expect((await load({ attr: "dark", systemDark: false })).initialTheme()).toBe("dark");
  });
  it("ignores a corrupt saved value", async () => {
    expect((await load({ stored: "purple", systemDark: true })).initialTheme()).toBe("dark");
  });
  it("still works when storage is blocked", async () => {
    const t = await load({ storageBroken: true, systemDark: true });
    expect(t.initialTheme()).toBe("dark");
    expect(() => t.applyTheme("light")).not.toThrow();
    expect(root().getAttribute("data-theme")).toBe("light");
  });
  it("still works with no matchMedia at all", async () => {
    vi.resetModules();
    vi.stubGlobal("matchMedia", undefined);
    expect((await import("./theme")).initialTheme()).toBe("light");
  });
});

describe("changing theme", () => {
  it("sets the attribute, remembers the choice and notifies listeners", async () => {
    const t = await load();
    const seen: string[] = [];
    function Probe() { seen.push(t.useTheme()); return null; }
    render(<Probe />);
    act(() => t.applyTheme("dark"));
    expect(root().getAttribute("data-theme")).toBe("dark");
    expect(localStorage.getItem("smartnest.theme")).toBe("dark");
    expect(seen.at(-1)).toBe("dark");
  });

  it("does not remember a system-driven start", async () => {
    const t = await load({ systemDark: true });
    t.startTheme();
    expect(root().getAttribute("data-theme")).toBe("dark");
    expect(localStorage.getItem("smartnest.theme")).toBeNull();      // only an explicit choice is saved
  });

  it("toggles back and forth", async () => {
    const t = await load({ attr: "light" });
    t.toggleTheme();
    expect(root().getAttribute("data-theme")).toBe("dark");
    t.toggleTheme();
    expect(root().getAttribute("data-theme")).toBe("light");
  });
});

describe("ThemeToggle", () => {
  it("names the mode it switches to and switches", async () => {
    const t = await load({ attr: "light" });
    t.startTheme();
    const { ThemeToggle } = await import("./components/ThemeToggle");
    const user = userEvent.setup();
    render(<ThemeToggle />);
    await user.click(screen.getByRole("button", { name: "Switch to dark mode" }));
    expect(root().getAttribute("data-theme")).toBe("dark");
    expect(screen.getByRole("button", { name: "Switch to light mode" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Switch to light mode" }));
    expect(root().getAttribute("data-theme")).toBe("light");
  });
});

describe("chart colours", () => {
  it("have a lighter palette and light text on dark, and never the same neutral colours as light", async () => {
    const { CHART } = await load();
    expect(CHART.dark.palette).toHaveLength(CHART.light.palette.length);
    expect(CHART.dark.palette).not.toEqual(CHART.light.palette);
    const lum = (hex: string) => {
      const n = parseInt(hex.slice(1), 16);
      return [(n >> 16) & 255, (n >> 8) & 255, n & 255].reduce((a, v, i) => a + v * [0.2126, 0.7152, 0.0722][i], 0);
    };
    expect(lum(CHART.dark.text)).toBeGreaterThan(lum(CHART.dark.surface) + 80);      // readable axis labels on the dark card
    expect(lum(CHART.light.text)).toBeLessThan(lum(CHART.light.surface) - 80);
    for (const key of ["text", "grid", "allocated", "surface", "ink"] as const) expect(CHART.dark[key]).not.toBe(CHART.light[key]);
  });

  it("every palette colour is bright enough to see on the dark card", async () => {
    const { CHART } = await load();
    const lum = (hex: string) => {
      const n = parseInt(hex.slice(1), 16);
      const f = (v: number) => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; };
      return 0.2126 * f((n >> 16) & 255) + 0.7152 * f((n >> 8) & 255) + 0.0722 * f(n & 255);
    };
    const bg = lum(CHART.dark.surface);
    for (const c of CHART.dark.palette) expect((lum(c) + 0.05) / (bg + 0.05)).toBeGreaterThanOrEqual(3);   // graphics need 3:1
  });
});
