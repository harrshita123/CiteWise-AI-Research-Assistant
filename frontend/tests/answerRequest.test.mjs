import assert from "node:assert/strict";
import test from "node:test";
import { matchesAnswerRequest } from "../src/utils/answerRequest.js";

const request = { requestId: "second", paperId: "paper-b", question: "Second question?",
  conversationId: "conversation-b", contextVersion: 2 };
const answer = { request_id: "second", paper_id: "paper-b", question: "Second question?",
  conversation_id: "conversation-b" };

test("accepts only the matching response", () => {
  assert.equal(matchesAnswerRequest(answer, request, "second", 2), true);
});

test("rejects an older response after a newer question or paper selection", () => {
  assert.equal(matchesAnswerRequest({ ...answer, request_id: "first" }, request, "second", 2), false);
  assert.equal(matchesAnswerRequest(answer, request, "first", 2), false);
  assert.equal(matchesAnswerRequest(answer, request, "second", 3), false);
  assert.equal(matchesAnswerRequest({ ...answer, paper_id: "paper-a" }, request, "second", 2), false);
});

test("rejects mismatched question and conversation", () => {
  assert.equal(matchesAnswerRequest({ ...answer, question: "Earlier question?" }, request, "second", 2), false);
  assert.equal(matchesAnswerRequest({ ...answer, conversation_id: "conversation-a" }, request, "second", 2), false);
});
