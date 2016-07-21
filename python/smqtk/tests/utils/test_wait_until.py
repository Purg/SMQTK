import threading
import time
import unittest

import nose.tools

from smqtk.utils import wait_until


class TestWaitUntil (unittest.TestCase):

    def test_neg_timeout(self):
        nose.tools.assert_raises(
            ValueError,
            wait_until, lambda: True, -1
        )

    def test_wait_success_immediate(self):
        nose.tools.assert_true(
            wait_until(lambda: True, 1)
        )

    def test_no_timeout(self):
        nose.tools.assert_true(
            wait_until(lambda: True, 0)
        )
        nose.tools.assert_false(
            wait_until(lambda: False, 0)
        )

    def test_delayed_success(self):
        e = threading.Event()
        fn = lambda: e.is_set()
        p = 0.5

        def set_n():
            time.sleep(p)
            e.set()

        t = threading.Thread(target=set_n)
        try:
            s = time.time()
            t.start()
            nose.tools.assert_true(
                wait_until(fn, 5.),
                "wait_until did not return True"
            )
            e = time.time()

            nose.tools.assert_true((e - s) > p,
                                   "wait_until should have taken as least %s "
                                   "seconds, took %s instead"
                                   % (p, e - s))
        finally:
            t.join()

    def test_timeout_failure(self):
        t = 1.2
        s = time.time()
        nose.tools.assert_false(
            wait_until(lambda: False, t)
        )
        e = time.time()
        nose.tools.assert_true(e - s, t)
