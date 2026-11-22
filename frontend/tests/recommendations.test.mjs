import test from "node:test";
import assert from "node:assert/strict";
import { recommendationView } from "../src/utils/recommendations.js";

test("recommendation states distinguish failure from an empty successful response", () => {
  assert.equal(recommendationView({ loading: true, error: false, count: 0 }), "loading");
  assert.equal(recommendationView({ loading: false, error: true, count: 0 }), "error");
  assert.equal(recommendationView({ loading: false, error: false, count: 0 }), "hidden");
  assert.equal(recommendationView({ loading: false, error: false, count: 3 }), "questions");
});
