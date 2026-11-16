import assert from "node:assert/strict";
import test from "node:test";
import { conversationViews } from "../src/utils/conversationViews.js";

const chat = (id, paper_id, updated_at, pinned = false) => ({
  id, paper_id, paper_filename: "same-name.pdf", title: id, updated_at, pinned, saved: true, messages: [],
});

test("pinned chats are global, folders use PDF IDs, and recents retain every chat once", () => {
  const records = [
    chat("older", "pdf-one", "2026-01-01T00:00:00Z", true),
    chat("newer", "pdf-two", "2026-01-03T00:00:00Z", true),
    chat("middle", "pdf-one", "2026-01-02T00:00:00Z"),
  ];
  const views = conversationViews(records);
  assert.deepEqual(views.recents.map((item) => item.id), ["newer", "middle", "older"]);
  assert.deepEqual(views.pinned.map((item) => item.id), ["newer", "older"]);
  assert.deepEqual([...views.folders.keys()], ["pdf-two", "pdf-one"]);
  assert.deepEqual(views.folders.get("pdf-one").map((item) => item.id), ["middle", "older"]);
  assert.equal(new Set(views.recents.map((item) => item.id)).size, records.length);
  assert.deepEqual(records.map((item) => item.id), ["older", "newer", "middle"]);
});

test("updates and deletion flow through every view of the same records", () => {
  const original = [chat("one", "pdf-one", "2026-01-01T00:00:00Z")];
  const changed = original.map((item) => item.id === "one" ? { ...item, title: "Renamed", pinned: true } : item);
  const views = conversationViews(changed);
  assert.equal(views.pinned[0].title, "Renamed");
  assert.equal(views.folders.get("pdf-one")[0], views.recents[0]);
  assert.equal(views.recents[0], views.pinned[0]);
  assert.equal(conversationViews(changed.filter((item) => item.id !== "one")).recents.length, 0);
});

test("zero, one, and multiple pins leave folders and recents intact", () => {
  const records = [chat("one", "pdf-one", "2026-01-01T00:00:00Z"),
    chat("two", "pdf-two", "2026-01-02T00:00:00Z")];
  assert.equal(conversationViews(records).pinned.length, 0);
  const onePinned = records.map((item) => item.id === "one" ? { ...item, pinned: true } : item);
  assert.deepEqual(conversationViews(onePinned).pinned.map((item) => item.id), ["one"]);
  const bothPinned = onePinned.map((item) => ({ ...item, pinned: true }));
  assert.deepEqual(conversationViews(bothPinned).pinned.map((item) => item.id), ["two", "one"]);
  const afterDelete = bothPinned.filter((item) => item.id !== "one");
  assert.deepEqual(conversationViews(afterDelete).pinned.map((item) => item.id), ["two"]);
  const afterLastDelete = afterDelete.filter((item) => item.id !== "two");
  assert.equal(conversationViews(afterLastDelete).pinned.length, 0);
  assert.equal(conversationViews(onePinned).recents.length, 2);
  assert.equal(conversationViews(onePinned).folders.size, 2);
});

test("saved and pinned views remain independent and share the same records", () => {
  const records = [chat("saved", "paper-one", "2026-01-01T00:00:00Z", false),
    { ...chat("pinned", "paper-two", "2026-01-02T00:00:00Z", true), saved: false }];
  const views = conversationViews(records);
  assert.deepEqual(views.saved.map((item) => item.id), ["saved"]);
  assert.deepEqual(views.pinned.map((item) => item.id), ["pinned"]);
  assert.equal(views.recents.length, 2);
  assert.equal(views.folders.size, 2);
  const changed = records.map((item) => item.id === "pinned" ? { ...item, saved: true } : item);
  assert.equal(conversationViews(changed).saved[0], conversationViews(changed).pinned[0]);
});

test("saved visibility follows only explicit persisted true values", () => {
  const first = chat("first", "pdf-1", "2026-01-01T00:00:00Z");
  const second = chat("second", "pdf-2", "2026-01-02T00:00:00Z");
  const legacy = { ...chat("legacy", "pdf-3", "2026-01-03T00:00:00Z") };
  delete legacy.saved;
  assert.deepEqual(conversationViews([]).saved, []);
  assert.deepEqual(conversationViews([first]).saved.map((item) => item.id), ["first"]);
  assert.deepEqual(conversationViews([first, second]).saved.map((item) => item.id), ["second", "first"]);
  assert.deepEqual(conversationViews([first, { ...second, saved: false }, legacy]).saved.map((item) => item.id), ["first"]);
  const noneSaved = [{ ...first, saved: false }, { ...second, saved: false }, legacy];
  assert.equal(conversationViews(noneSaved).saved.length, 0);
  assert.equal(conversationViews(noneSaved).recents.length, 3);
});

test("unsave, unpin, and delete affect only their own views", () => {
  const both = chat("both", "pdf-1", "2026-01-02T00:00:00Z", true);
  const saved = chat("saved", "pdf-2", "2026-01-01T00:00:00Z");
  const unsaved = [{ ...both, saved: false }, saved];
  assert.deepEqual(conversationViews(unsaved).saved.map((item) => item.id), ["saved"]);
  assert.deepEqual(conversationViews(unsaved).pinned.map((item) => item.id), ["both"]);
  const unpinned = [{ ...both, pinned: false }, saved];
  assert.deepEqual(conversationViews(unpinned).saved.map((item) => item.id), ["both", "saved"]);
  assert.equal(conversationViews(unpinned).pinned.length, 0);
  const deleted = [saved];
  assert.equal(conversationViews(deleted).recents.length, 1);
  assert.equal(conversationViews(deleted).folders.get("pdf-1"), undefined);
});

test("PDF folders include ready documents without conversations", () => {
  const papers = [{ id: "pdf-empty", filename: "empty.pdf" }, { id: "pdf-used", filename: "used.pdf" }];
  const views = conversationViews([chat("one", "pdf-used", "2026-01-01T00:00:00Z")], papers);
  assert.deepEqual([...views.folders.keys()], ["pdf-empty", "pdf-used"]);
  assert.equal(views.folders.get("pdf-empty").length, 0);
  assert.equal(views.folders.get("pdf-used").length, 1);
});
