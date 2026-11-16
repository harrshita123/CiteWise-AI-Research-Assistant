import assert from "node:assert/strict";
import test from "node:test";
import { applyTheme, readTheme } from "../src/utils/theme.js";

test("theme defaults to dark and restores a saved light preference", () => {
  const values = new Map();
  const storage = { getItem: (key) => values.get(key) ?? null, setItem: (key, value) => values.set(key, value) };
  assert.equal(readTheme(storage), "dark");
  const root = { dataset: {} };
  applyTheme("light", root, storage);
  assert.equal(root.dataset.theme, "light");
  assert.equal(readTheme(storage), "light");
  applyTheme("dark", root, storage);
  assert.equal(readTheme(storage), "dark");
});
