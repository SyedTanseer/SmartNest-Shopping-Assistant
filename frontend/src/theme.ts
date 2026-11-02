import { useSyncExternalStore } from "react";

export type Theme = "light" | "dark";

const KEY = "smartnest.theme";
const listeners = new Set<() => void>();

function stored(): Theme | null {
  try {
    const v = localStorage.getItem(KEY);
    return v === "light" || v === "dark" ? v : null;
  } catch {
    return null;                       // private mode / blocked storage
  }
}

function system(): Theme {
  try {
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  } catch {
    return "light";
  }
}

/** The saved choice, else the system setting.  (index.html sets the attribute before the first paint.) */
export function initialTheme(): Theme {
  const attr = typeof document !== "undefined" ? document.documentElement.getAttribute("data-theme") : null;
  return attr === "light" || attr === "dark" ? attr : stored() ?? system();
}

let current: Theme = initialTheme();

export function applyTheme(theme: Theme, persist = true): void {
  current = theme;
  document.documentElement.setAttribute("data-theme", theme);
  if (persist) {
    try { localStorage.setItem(KEY, theme); } catch { /* the choice just will not be remembered */ }
  }
  listeners.forEach((l) => l());
}

export const toggleTheme = (): void => applyTheme(current === "dark" ? "light" : "dark");

// Follow the system setting for as long as the user has not chosen explicitly.
try {
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (e) => {
    if (!stored()) applyTheme(e.matches ? "dark" : "light", false);
  });
} catch { /* no matchMedia (tests) */ }

export function startTheme(): void {
  applyTheme(initialTheme(), false);
}

const subscribe = (l: () => void) => {
  listeners.add(l);
  return () => { listeners.delete(l); };
};

export const useTheme = (): Theme => useSyncExternalStore(subscribe, () => current, () => "light" as Theme);

/** Colours for the charts (Recharts takes literal colours, not CSS variables).  Same palette family as the tokens. */
export const CHART = {
  light: {
    palette: ["#0f766e", "#d97706", "#4f46e5", "#e11d48", "#0284c7", "#65a30d", "#7c3aed"],
    text: "#6b6560", grid: "#e7e2db", allocated: "#cbd5d1", spent: "#0f766e", surface: "#ffffff", ink: "#1c1917",
  },
  dark: {
    palette: ["#4fd1c0", "#f0b866", "#9aa5ff", "#fb7a95", "#5cc0f5", "#b4e05a", "#b79bff"],
    text: "#a59e92", grid: "#302c25", allocated: "#3b362d", spent: "#5ccfbf", surface: "#1a1815", ink: "#ece7de",
  },
} as const;

export const useChartColors = () => CHART[useTheme()];
