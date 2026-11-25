import test from "node:test";
import assert from "node:assert/strict";
import { recommendationView, researchQuestionMode } from "../src/utils/recommendations.js";

test("recommendation states distinguish failure from an empty successful response", () => {
  assert.equal(recommendationView({ loading: true, error: false, count: 0 }), "loading");
  assert.equal(recommendationView({ loading: false, error: true, count: 0 }), "error");
  assert.equal(recommendationView({ loading: false, error: false, count: 0 }), "hidden");
  assert.equal(recommendationView({ loading: false, error: false, count: 3 }), "questions");
});

test("switching demo papers changes suggestion clicks between sample answers and passage search", () => {
  const demo = { demoMode: true, demoSample: true, canAnswer: false, retryRemaining: 30 };
  assert.equal(researchQuestionMode(demo), "ask");
  assert.equal(researchQuestionMode({ ...demo, demoSample: false, canAnswer: true }), "search");
  assert.equal(researchQuestionMode(demo), "ask");
});

test("production suggestion clicks still respect provider availability and cooldown", () => {
  const live = { demoMode: false, demoSample: false, canAnswer: true, retryRemaining: 0 };
  assert.equal(researchQuestionMode(live), "ask");
  assert.equal(researchQuestionMode({ ...live, retryRemaining: 30 }), "search");
  assert.equal(researchQuestionMode({ ...live, canAnswer: false }), "search");
});
