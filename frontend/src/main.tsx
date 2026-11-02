import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./index.css";
import { startTheme } from "./theme";

startTheme();
requestAnimationFrame(() => requestAnimationFrame(() => document.documentElement.classList.add("theme-ready")));

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
