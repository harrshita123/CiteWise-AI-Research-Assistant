import assert from "node:assert/strict";
import test from "node:test";
import { deadlineFromRetry, remainingSeconds } from "../src/utils/groqCooldown.js";

test("a successful answer clears an active retry deadline", () => {
  const active = deadlineFromRetry(90, 1000);
  assert.equal(remainingSeconds(active, 1000), 90);
  assert.equal(remainingSeconds(0, 1000), 0);
});

test("a changed key makes the server report no remaining cooldown", () => {
  const oldDeadline = deadlineFromRetry(90, 1000);
  const replacement = deadlineFromRetry(0, 2000);
  assert.equal(remainingSeconds(oldDeadline, 2000), 89);
  assert.equal(replacement, 0);
});

test("an unchanged key retains its active provider retry time after refresh", () => {
  const restored = deadlineFromRetry(75, 2000);
  assert.equal(remainingSeconds(restored, 2000), 75);
  assert.equal(remainingSeconds(restored, 77000), 0);
});

test("missing retry timing never creates an invented countdown", () => {
  assert.equal(deadlineFromRetry(null, 1000), 0);
  assert.equal(deadlineFromRetry(Number.NaN, 1000), 0);
});
