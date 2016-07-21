"""
LICENCE
-------
Copyright 2013-2016 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import logging
import operator as op
import time


class SmqtkObject (object):
    """
    Highest level object interface for classes defined in SMQTK.

    Currently defines logging methods.

    """

    @classmethod
    def logger(cls):
        """
        :return: logging object for this class
        :rtype: logging.Logger
        """
        return logging.getLogger('.'.join((cls.__module__, cls.__name__)))

    @property
    def _log(self):
        """
        :return: logging object for this class as a property
        :rtype: logging.Logger
        """
        return self.logger()


def ncr(n, r):
    """
    N-choose-r method, returning the number of combinations possible in integer
    form.

    From dheerosaur:
        http://stackoverflow.com/questions/4941753/is-there-a-math-ncr-function-in-python

    :param n: Selection pool size.
    :type n: int

    :param r: permutation selection size.
    :type r: int

    :return: Number of n-choose-r permutations for the given n and r.
    :rtype: int

    """
    r = min(r, n - r)
    if r == 0:
        return 1
    numer = reduce(op.mul, xrange(n, n - r, -1))
    denom = reduce(op.mul, xrange(1, r + 1))
    return numer // denom


def merge_dict(a, b):
    """
    Merge dictionary b into dictionary a. Congruent keys are overwritten,
    while new keys are added.

    This is different than normal dictionary update in that we don't bash
    nested dictionaries, instead recursively updating them.

    :param a: The "base" dictionary.
    :type a: dict

    :param b: The dictionary to merge into ``a`` recursively.
    :type b: dict

    """
    for k in b:
        if k in a and isinstance(a[k], dict) and isinstance(b[k], dict):
            merge_dict(a[k], b[k])
        else:
            a[k] = b[k]


def wait_until(fn, timeout, period=0.01, *args, **kwargs):
    """
    Wait until the given function ``fn`` returns a true-evaluating value or the
    timeout expires.

    ``fn`` is executed at least once even if timeout is 0.

    Positional and keyword arguments after this function's defined ones are
    passed to the function ``fn``.

    This function may take longer than ``timeout`` seconds to complete if ``fn``
    is a long running function. Ideally, this function is used with a ``fn``
    that is short-running or is checking the value of a variable that is
    externally modified. E.g. thread alive status, thread/process updated value,
    etc.

    :param fn: function to evaluate
    :param timeout: The time to allow fn to return a true-evaluating value in
        seconds.
    :param period: The interval at which to check ``fn``

    :return: True if ``fn`` returned a true-evaluating value before the time-out
        period. False otherwise.

    """
    if timeout < 0:
        raise ValueError("Timeout must be >0.")
    e = time.time() + timeout
    while True:
        r = fn(*args, **kwargs)
        if bool(r):
            return r
        elif time.time() >= e:
            return False


###
# In specific ordering for dependency resolution
#

# No internal util dependencies
from .bin_utils import initialize_logging
from .configurable_interface import Configurable
from .database_info import DatabaseInfo
from .read_write_lock import ReaderUpdateException, DummyRWLock, ReadWriteLock
from .safe_config_comment_parser import SafeConfigCommentParser
from .signal_handler import SignalHandler
from .simple_timer import SimpleTimer
