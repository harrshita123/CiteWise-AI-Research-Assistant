const THEME_KEY = "citewise-theme";

export function readTheme(storage = globalThis.localStorage) {
  try { return storage.getItem(THEME_KEY) === "light" ? "light" : "dark"; }
  catch { return "dark"; }
}

export function applyTheme(theme, root = document.documentElement, storage = globalThis.localStorage) {
  root.dataset.theme = theme;
  try { storage.setItem(THEME_KEY, theme); }
  catch { /* The current page still switches theme when storage is unavailable. */ }
}
