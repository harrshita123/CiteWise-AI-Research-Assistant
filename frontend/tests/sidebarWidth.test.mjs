import assert from "node:assert/strict";
import test from "node:test";
import { clampSidebarWidth, DEFAULT_SIDEBAR_WIDTH, resizedSidebarWidth } from "../src/utils/sidebarWidth.js";

test("expanded sidebar width is bounded and can be restored from storage", () => {
  assert.equal(clampSidebarWidth(undefined), DEFAULT_SIDEBAR_WIDTH);
  assert.equal(DEFAULT_SIDEBAR_WIDTH, 300);
  assert.equal(clampSidebarWidth("350"), 350);
  assert.equal(clampSidebarWidth(100), 220);
  assert.equal(clampSidebarWidth(600), 420);
  assert.equal(clampSidebarWidth("invalid"), DEFAULT_SIDEBAR_WIDTH);
});

test("dragging in either direction follows the pointer and respects limits", () => {
  assert.equal(resizedSidebarWidth(300, 300, 350), 350);
  assert.equal(resizedSidebarWidth(300, 300, 260), 260);
  assert.equal(resizedSidebarWidth(300, 300, 100), 220);
  assert.equal(resizedSidebarWidth(300, 300, 600), 420);
});
