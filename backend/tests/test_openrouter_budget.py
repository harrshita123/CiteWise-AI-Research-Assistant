"""Integration test for atomic free-account admission across MongoDB clients."""

import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from uuid import uuid4

from pymongo import MongoClient

from app.providers.answers import AnswerRateLimitError
from app.providers.openrouter_routing import SharedRequestBudget, UpstreamPoolLimit


@unittest.skipUnless(os.getenv("CITEWISE_TEST_MONGO_URI"), "isolated MongoDB URI not configured")
class SharedRequestBudgetTests(unittest.TestCase):
    def test_expired_undispatched_ticket_is_reconciled_but_legacy_and_uncertain_usage_count(self):
        uri = os.environ["CITEWISE_TEST_MONGO_URI"]
        database = f"citewise_budget_test_{uuid4().hex}"
        with MongoClient(uri, serverSelectionTimeoutMS=2000) as first, \
                MongoClient(uri, serverSelectionTimeoutMS=2000) as second:
            try:
                collection = first[database].requests
                collection.insert_one({"_id": "free", "attempts": [{"ticket": "legacy", "at": 99999}]})
                left = SharedRequestBudget(collection, "free", 3, 3)
                right = SharedRequestBudget(second[database].requests, "free", 3, 3)
                with patch("app.providers.openrouter_routing.time.time", return_value=100000) as clock:
                    uncertain = left.reserve("primary", stage="assessment", attempt=1)
                    self.assertTrue(left.dispatch(uncertain))
                    left.finish(uncertain, "transport_failed", error_type="ReadTimeout")
                    abandoned = left.reserve("primary", stage="generation", attempt=1)
                    with self.assertRaises(AnswerRateLimitError) as blocked:
                        right.reserve("fallback")
                    self.assertEqual(blocked.exception.retry_after, str(left.reservation_seconds))
                    clock.return_value += left.reservation_seconds
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        stale = pool.submit(left.dispatch, abandoned)
                        admitted = pool.submit(right.reserve, "fallback", stage="generation", attempt=2)
                        self.assertFalse(stale.result())
                        replacement = admitted.result()
                    self.assertTrue(right.dispatch(replacement))
                    right.finish(replacement, "provider_rejected", http_status=429)
                    left.finish(replacement, "http_success")

                record = collection.find_one({"_id": "free"})
                states = {item["ticket"]: item.get("state") for item in record["attempts"]}
                self.assertEqual(states, {"legacy": None, uncertain: "unknown", replacement: "completed"})
                self.assertEqual(record["attempts"][-1]["outcome"], "provider_rejected")
                self.assertEqual([(item["ticket"], item["state"]) for item in record["released_reservations"]],
                                 [(abandoned, "failed_before_dispatch")])
                immediate = SharedRequestBudget(collection, "immediate", 1, 1)
                ticket = immediate.reserve()
                self.assertTrue(immediate.fail_before_dispatch(ticket))
                self.assertFalse(immediate.dispatch(ticket))
                self.assertNotEqual(immediate.reserve(), ticket)
            finally:
                first.drop_database(database)

    def test_predispatch_release_racing_dispatch_keeps_exactly_one_transition(self):
        uri = os.environ["CITEWISE_TEST_MONGO_URI"]
        database = f"citewise_budget_test_{uuid4().hex}"
        with MongoClient(uri, serverSelectionTimeoutMS=2000) as first, \
                MongoClient(uri, serverSelectionTimeoutMS=2000) as second:
            try:
                collection = first[database].requests
                left = SharedRequestBudget(collection, "free", 1, 1)
                right = SharedRequestBudget(second[database].requests, "free", 1, 1)
                ticket = left.reserve("primary", key_id="sha256:audit-a")
                initial = collection.find_one({"_id": "free"})["attempts"][0]
                self.assertEqual((initial["state"], initial["account_scope"], initial["key_id"]),
                                 ("reserved", "free", "sha256:audit-a"))
                with ThreadPoolExecutor(max_workers=2) as pool:
                    release = pool.submit(left.fail_before_dispatch, ticket)
                    dispatch = pool.submit(right.dispatch, ticket)
                    released, dispatched = release.result(), dispatch.result()
                self.assertNotEqual(released, dispatched)
                if dispatched:
                    right.finish(ticket, "transport_failed", error_type="ReadTimeout")
                    with self.assertRaises(AnswerRateLimitError):
                        left.reserve("alternate", key_id="sha256:audit-b")
                    self.assertFalse(left.fail_before_dispatch(ticket))
                    attempt = collection.find_one({"_id": "free"})["attempts"][0]
                    self.assertEqual((attempt["state"], attempt["outcome"]), ("unknown", "transport_failed"))
                else:
                    replacement = left.reserve("alternate", key_id="sha256:audit-b")
                    self.assertTrue(left.dispatch(replacement))
                    self.assertFalse(right.dispatch(ticket))
                    self.assertFalse(right.fail_before_dispatch(ticket))
                    record = collection.find_one({"_id": "free"})
                    self.assertEqual(record["released_reservations"][0]["outcome"], "dispatch_not_started")
                    self.assertEqual(len(record["attempts"]), 1)
            finally:
                first.drop_database(database)

    def test_reconciliation_preserves_records_without_positive_pre_dispatch_evidence(self):
        uri = os.environ["CITEWISE_TEST_MONGO_URI"]
        database = f"citewise_budget_test_{uuid4().hex}"
        with MongoClient(uri, serverSelectionTimeoutMS=2000) as client:
            try:
                collection = client[database].requests
                entries = [
                    {"ticket": "legacy", "at": 99999},
                    {"ticket": "missing_expiration", "at": 99998, "state": "reserved"},
                    {"ticket": "dispatch_marker", "at": 99997, "state": "reserved",
                     "expires_at": 99999, "dispatched_at": 99997},
                ]
                collection.insert_one({"_id": "free", "attempts": entries})
                budget = SharedRequestBudget(collection, "free", 3, 3)
                with patch("app.providers.openrouter_routing.time.time", return_value=100000):
                    with self.assertRaises(AnswerRateLimitError) as blocked:
                        budget.reserve()
                row = collection.find_one({"_id": "free"})
                self.assertEqual(blocked.exception.limit_source, "local_daily_budget")
                self.assertEqual(row["attempts"], entries)
                self.assertEqual(row["released_reservations"], [])
            finally:
                client.drop_database(database)

    def test_parallel_workers_share_limits_and_cooldown(self):
        uri = os.environ["CITEWISE_TEST_MONGO_URI"]
        database = f"citewise_budget_test_{uuid4().hex}"
        with MongoClient(uri, serverSelectionTimeoutMS=2000) as first, \
                MongoClient(uri, serverSelectionTimeoutMS=2000) as second:
            try:
                clients = [first, second]
                budgets = [SharedRequestBudget(client[database].requests, "free", 3, 2)
                           for client in clients]

                def reserve(index):
                    try:
                        budget = budgets[index % 2]
                        return budget.dispatch(budget.reserve())
                    except AnswerRateLimitError:
                        return False

                with ThreadPoolExecutor(max_workers=8) as pool:
                    self.assertEqual(sum(pool.map(reserve, range(8))), 2)

                fresh = SharedRequestBudget(second[database].requests, "free", 3, 3)
                fresh.dispatch(fresh.reserve())
                with self.assertRaises(AnswerRateLimitError) as daily:
                    fresh.reserve()
                self.assertGreater(int(daily.exception.retry_after), 86000)
                self.assertEqual(daily.exception.limit_source, "local_daily_budget")

                cooldown = [SharedRequestBudget(client[database].requests, "other") for client in clients]
                cooldown[0].reserve()
                cooldown[0].cool_down(90, limit_source="openrouter_free_tier_daily")
                cooldown[1].cool_down(20)
                with self.assertRaises(AnswerRateLimitError) as blocked:
                    cooldown[1].reserve()
                self.assertGreater(int(blocked.exception.retry_after), 80)
                self.assertEqual(blocked.exception.limit_source, "openrouter_free_tier_daily")
                scoped = [SharedRequestBudget(client[database].requests, "scoped", 3, 2) for client in clients]
                scoped[0].reserve("primary")
                scoped[0].cool_down(60, model="primary")
                with self.assertRaises(UpstreamPoolLimit) as upstream_limit:
                    scoped[1].reserve("primary")
                self.assertEqual(upstream_limit.exception.limit_source, "upstream_provider_shared_pool")
                scoped[1].reserve("alternate")
                with self.assertRaises(AnswerRateLimitError) as account_limit:
                    scoped[0].reserve("primary")
                self.assertNotIsInstance(account_limit.exception, UpstreamPoolLimit)
                self.assertEqual(account_limit.exception.limit_source, "local_minute_budget")
                attempts = first[database].requests.find_one({"_id": "scoped"})["attempts"]
                self.assertEqual(len(attempts), 2)
            finally:
                first.drop_database(database)
