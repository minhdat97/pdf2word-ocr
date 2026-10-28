"""The task queue (app/tasks.py): at most CAREFUL_WORKERS conversions with the formula model ("Đọc kỹ") run at
once; a careful task that waits stays "queued" without holding a thread, the tasks behind it go ahead; the queue
and per-file limits hold. Conversions are fakes that wait for the test to let them finish.

    python -m unittest discover -s tests
"""

import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import tasks  # noqa: E402

WAIT = 5  # seconds; only reached when a test fails


class FakeConversions:
    """convert() stand-in: each call says it started and waits until the test releases it; records how many
    careful conversions ran at the same time."""

    def __init__(self):
        self.started: dict[str, threading.Event] = {}
        self.release: dict[str, threading.Event] = {}
        self.lock = threading.Lock()
        self.careful_now = self.careful_most = 0

    def events(self, name: str) -> tuple[threading.Event, threading.Event]:
        with self.lock:
            return self.started.setdefault(name, threading.Event()), self.release.setdefault(name, threading.Event())

    def __call__(self, path, mode, languages, progress, strip=False, careful=False):
        started, release = self.events(Path(path).name)
        with self.lock:
            self.careful_now += careful
            self.careful_most = max(self.careful_most, self.careful_now)
        started.set()
        release.wait(WAIT)
        with self.lock:
            self.careful_now -= careful
        return Path(path), {"careful": careful}


class QueueTest(unittest.TestCase):
    def setUp(self):
        self.fake = FakeConversions()
        self.n = 0
        patches = [
            mock.patch.object(tasks, "convert", self.fake),
            mock.patch.object(tasks.storage, "exists", return_value=True),
            mock.patch.object(tasks.storage, "path_for", side_effect=lambda token: Path(token)),
            mock.patch.object(tasks.storage, "store_output", return_value="out.docx"),
            mock.patch.object(tasks, "available_languages", return_value=["vie"]),
            mock.patch.object(tasks.mathocr, "available", return_value=True),
            mock.patch.object(tasks, "FORMULAS", "scripts"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.manager = tasks.TaskManager(workers=2, careful_workers=1)
        self.addCleanup(self.finish)

    def finish(self):
        for release in list(self.fake.release.values()):
            release.set()
        self.manager.shutdown()
        self.manager._pool.shutdown(wait=True)  # before the patches go

    def submit(self, careful: bool = False, ocr_type: str = "ocr", token: str | None = None) -> tasks.Task:
        self.n += 1
        token = token or f"f{self.n}.pdf"
        self.fake.events(token)
        return self.manager.submit({
            "task_id": f"{self.n:032x}", "tool": "pdf-to-word", "input_tokens": [token],
            "options": {"ocr_type": ocr_type, "languages": ["vie"], "careful": careful},
        })

    def started(self, task: tasks.Task) -> bool:
        return self.fake.started[task.input_tokens[0]].wait(WAIT)

    def finish_task(self, task: tasks.Task) -> None:
        self.fake.release[task.input_tokens[0]].set()

    def waiting(self, task: tasks.Task) -> bool:
        return task.status == "queued" and not self.fake.started[task.input_tokens[0]].is_set()

    def test_one_careful_at_a_time_without_blocking_the_others(self):
        a = self.submit(careful=True)
        self.assertTrue(self.started(a))
        b = self.submit(careful=True)  # waits for a: no thread taken
        c = self.submit()  # takes the free thread, ahead of b
        self.assertTrue(self.started(c))  # (b would have had it, being older, had it been allowed to run)
        self.assertTrue(self.waiting(b))
        self.assertEqual(a.status, "processing")

        self.finish_task(c)  # the free thread does not go to b while a runs
        d = self.submit()
        self.assertTrue(self.started(d))
        self.assertTrue(self.waiting(b))

        self.finish_task(a)  # now b runs
        self.assertTrue(self.started(b))
        self.assertEqual(self.fake.careful_most, 1)
        for t in (b, d):
            self.finish_task(t)

    def test_careful_waits_for_a_thread_in_turn(self):
        a, b = self.submit(), self.submit()  # both threads busy
        self.assertTrue(self.started(a) and self.started(b))
        c = self.submit(careful=True)
        d = self.submit()
        self.finish_task(a)  # the oldest waiting task (c) gets the thread
        self.assertTrue(self.started(c))
        self.assertTrue(self.waiting(d))
        self.finish_task(b)
        self.assertTrue(self.started(d))

    def test_careful_without_the_model_is_not_held(self):
        with mock.patch.object(tasks.mathocr, "available", return_value=False):
            a, b = self.submit(careful=True), self.submit(careful=True)
        self.assertTrue(self.started(a) and self.started(b))  # both at once: the model is not used
        c, d = self.submit(careful=True, ocr_type=""), self.submit()  # basic mode reads no scan
        self.finish_task(a)
        self.finish_task(b)
        self.assertTrue(self.started(c) and self.started(d))

    def test_status_and_result(self):
        a = self.submit(careful=True)
        self.assertTrue(self.started(a))
        b = self.submit(careful=True)
        self.assertEqual(b.to_json()["status"], "queued")
        self.assertIsNone(b.to_json()["result"])
        self.finish_task(a)
        self.assertTrue(self.started(b))
        self.finish_task(b)
        for t in (a, b):
            for _ in range(WAIT * 100):
                if t.status == "done":
                    break
                threading.Event().wait(0.01)
            self.assertEqual(t.status, "done")
            self.assertEqual(t.output_files, ["out.docx"])
            self.assertIs(t.result_options["careful"], True)

    def test_limits(self):
        with mock.patch.object(tasks, "MAX_PENDING_TASKS", 3):
            a = self.submit(careful=True)
            self.submit(careful=True)
            self.submit()
            with self.assertRaises(tasks.Busy) as busy:
                self.submit()
            self.assertEqual(busy.exception.code, "busy")
        with self.assertRaises(tasks.Busy) as same:  # this file already has a task running
            self.submit(token=a.input_tokens[0])
        self.assertEqual(same.exception.code, "too_many_tasks")

    def test_one_careful_runs_alone_when_the_limit_is_one_worker(self):
        manager = tasks.TaskManager(workers=1, careful_workers=1)
        self.manager.shutdown()
        self.manager = manager
        a = self.submit(careful=True)
        b = self.submit()
        self.assertTrue(self.started(a))
        self.assertTrue(self.waiting(b))
        self.finish_task(a)
        self.assertTrue(self.started(b))


class AnalyseResultTest(unittest.TestCase):
    def test_careful_info(self):
        with mock.patch.object(tasks.mathocr, "available", return_value=True), \
                mock.patch.object(tasks, "FORMULAS", "scripts"), mock.patch.object(tasks, "MFR_MAX_PAGES", 5):
            self.assertEqual(tasks.careful_info(), {"available": True, "max_scan_pages": 5})
        with mock.patch.object(tasks.mathocr, "available", return_value=False):
            self.assertFalse(tasks.careful_info()["available"])
        with mock.patch.object(tasks.mathocr, "available", return_value=True), mock.patch.object(tasks, "FORMULAS", "model"):
            self.assertFalse(tasks.careful_info()["available"])  # every conversion is careful already


if __name__ == "__main__":
    unittest.main()
