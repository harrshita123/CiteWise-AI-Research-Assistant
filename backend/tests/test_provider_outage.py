"""Opt-in destructive restart of a disposable mongod owned by this test."""
import multiprocessing
import os
import shutil
import socket
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from pymongo import MongoClient
from app.services.paper_discovery import AcademicClient, ProviderError
from app.services.provider_coordination import ProviderCoordinator


def worker(uri, old, begin, entered, finish, results, renewal_failed):
    with MongoClient(uri, serverSelectionTimeoutMS=250, connectTimeoutMS=250) as client:
        coord = ProviderCoordinator(client.outage.providers, lease_seconds=1)
        renew = coord.renew
        def observed_renew(provider, token):
            try:
                return renew(provider, token)
            except Exception:
                renewal_failed.set()
                raise
        coord.renew = observed_renew
        api = AcademicClient(coordinator=coord)
        begin.wait(20)
        def transport(*args):
            entered.set()
            if old:
                finish.wait(20)
            return b'{"message":{"title":["old"]}}' if old else b'{"message":{"title":["new"]}}'
        try:
            with patch.object(api, "_get_local", side_effect=transport):
                data = api.get("crossref", "/works/10.1234%2Fexample")
            results.put(data)
        except ProviderError as exc:
            results.put(str(exc))


@unittest.skipUnless(os.getenv("CITEWISE_RUN_OUTAGE_TEST") and shutil.which("mongod"), "requires isolated local mongod")
class OutageTests(unittest.TestCase):
    def test_active_request_during_database_restart(self):
        ctx = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory() as directory, socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            sock.close()
            uri = f"mongodb://127.0.0.1:{port}"
            def start():
                return subprocess.Popen(["mongod", "--dbpath", directory, "--port", str(port),
                    "--bind_ip", "127.0.0.1", "--nounixsocket", "--logpath", str(Path(directory)/"mongo.log"), "--logappend"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            db = start()
            begins = [ctx.Event(), ctx.Event()]
            entered = [ctx.Event(), ctx.Event()]
            finish, results = ctx.Event(), ctx.Queue()
            renewal_failed = ctx.Event()
            workers = []
            try:
                with MongoClient(uri, serverSelectionTimeoutMS=5000) as client:
                    client.admin.command("ping")
                    for i in range(2):
                        proc = ctx.Process(target=worker, args=(uri, i == 0, begins[i], entered[i], finish, results, renewal_failed))
                        proc.start()
                        workers.append(proc)
                    begins[0].set()
                    self.assertTrue(entered[0].wait(10))
                    db.terminate()
                    db.wait(15)
                    self.assertTrue(renewal_failed.wait(5))
                    time.sleep(1.1)
                    db = start()
                    client.admin.command("ping")
                    begins[1].set()
                    self.assertTrue(entered[1].wait(10))
                    self.assertIn(b"new", results.get(timeout=10))
                    finish.set()
                    self.assertEqual(results.get(timeout=10), "connection error")
                    api = AcademicClient(coordinator=ProviderCoordinator(client.outage.providers))
                    with patch.object(api, "_get_local", side_effect=AssertionError("cache miss")):
                        self.assertIn(b"new", api.get("crossref", "/works/DOI:10.1234/EXAMPLE"))
            finally:
                finish.set()
                for proc in workers:
                    proc.join(2)
                    if proc.is_alive():
                        proc.kill()
                        proc.join()
                db.terminate()
                db.wait(15)
