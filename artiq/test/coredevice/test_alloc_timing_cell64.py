"""Allocation latency under traditional reference counting vs. CTRC, sized for
64-byte CTRC cells.

This is ``test_alloc_timing.py`` adapted to a NAC3 built with
``CTRC_CELL_SIZE = 64`` (NAC3's ``experiment-cell64`` branch). Run it against
that build; the 128-byte file does not compile there.

Every object built here fits in one 64-byte CTRC cell on the 32-bit core
device. A list object is 16 B and a ``_Point`` is 16 B. A list buffer is
16 B + 4 B per slot, rounded up to a multiple of 20 B, so it fits in a cell
with at most 11 slots; a comprehension over ``range(n)`` allocates ``n + 1``
slots, so it may have at most 10 elements.
Differences from the 128-byte file:

* ``burst`` builds 10 inner lists instead of 24, so its outer buffer has 11
  slots: 16 + 11 * 4 = 60 B.
* There is no ``exception`` workload. An exception object is 72 B, which no
  longer fits in a cell, so ``raise`` inside ``with critical(...)`` is rejected.
  The RC cost of raising is unchanged from the 128-byte file.
* ``_CTRC_PAGES`` is 32: 32 * 63 = 2016 cells, at least the 1984 objects that
  64 pages held with 128-byte cells.

``none`` has an empty timed region; subtract it from every other row.

    python -m unittest -v artiq.test.coredevice.test_alloc_timing_cell64
"""

import numpy
from numpy import int32

from artiq.experiment import *
from artiq.coredevice.core import Core
from artiq.test.hardware_testbench import ExperimentCase


WL_NONE = 0
WL_LIST = 1
WL_OBJECT = 2
WL_TUPLE_OF_LISTS = 3
WL_LIST_OF_LISTS = 4
WL_BURST = 5
WL_RANGE = 6

# Pages reserved by ``with critical(...)``. One page is 63 cells. The largest
# workload (``burst``) keeps 1 outer + 10 inner lists, 2 cells each, plus one
# range object live at a time.
_CTRC_PAGES = 32


@compile
class _PointCell64:
    x: Kernel[int32]
    y: Kernel[int32]

    @kernel
    def __init__(self, x: int32, y: int32):
        self.x = x
        self.y = y


@compile
class _AllocTimingCell64(EnvExperiment):
    core: KernelInvariant[Core]
    n: KernelInvariant[int32]
    ts: Kernel[list[float]]
    acc: Kernel[int32]

    def build(self, n=1000):
        self.setattr_device("core")
        self.n = n
        self.ts = [0.0] * n
        self.acc = 0

    @kernel
    def _loop(self, which: int32):
        acc = int32(0)
        for i in range(self.n):
            t0 = self.core.get_rtio_counter_mu()
            if which == WL_LIST:
                # 2 objects: list + buffer (32 B)
                a = [i, i + 1, i + 2, i + 3]
                acc += a[0]
            elif which == WL_OBJECT:
                # 1 leaf object
                p = _PointCell64(i, i)
                acc += p.x
            elif which == WL_TUPLE_OF_LISTS:
                # 4 objects; the tuple itself is inline on the stack, dropping
                # it walks both lists
                t = ([i], [i + 1])
                acc += t[0][0]
            elif which == WL_LIST_OF_LISTS:
                # 8 objects, drop recursion depth 2
                l = [[i], [i + 1], [i + 2]]
                acc += l[2][0]
            elif which == WL_BURST:
                # 1 outer (16 B + 11 slots = 60 B buffer) + 10 inner lists,
                # plus one range object: ~23 cells live, all dropped at once
                b = [[i] for _ in range(10)]
                acc += b[9][0]
            elif which == WL_RANGE:
                # 1 leaf object
                for j in range(i, i + 2):
                    acc += j
            t1 = self.core.get_rtio_counter_mu()
            self.ts[i] = self.core.mu_to_seconds(t1 - t0)
        self.acc = acc

    # One entry point per mode. NAC3 makes a kernel reserve CTRC pages on entry
    # if a ``with critical`` block is reachable from it, whatever the branch
    # conditions on the way, so the RC kernel must not reach one at all.

    @kernel
    def bench_rc(self, which: int32):
        self._loop(which)

    @kernel
    def bench_ctrc(self, which: int32):
        with critical(_CTRC_PAGES):
            self._loop(which)


class _AllocTimingCell64Mixin:
    """Test bodies shared by the RC and CTRC classes. Not a TestCase itself."""

    ctrc = False
    n = 1000

    @classmethod
    def setUpClass(cls):
        cls.results = []

    @classmethod
    def tearDownClass(cls):
        if not cls.results:
            return
        width = max(len(r[0]) for r in cls.results)
        print()
        print("{} (n={}, ctrc={}, 64 B cells)".format(cls.__name__, cls.n, cls.ctrc))
        print("| {} | mean (us) |  std (us) |  max (us) |".format("workload".ljust(width)))
        print("| {} | --------- | --------- | --------- |".format("-" * width))
        for name, mean, std, mx in cls.results:
            print("| {} | {:>9.3f} | {:>9.3f} | {:>9.3f} |".format(
                name.ljust(width), mean * 1e6, std * 1e6, mx * 1e6))

    def _bench(self, name, which):
        exp = self.create(_AllocTimingCell64, n=self.n)
        if self.ctrc:
            exp.bench_ctrc(which)
        else:
            exp.bench_rc(which)
        ts = numpy.array(exp.ts)
        self.results.append((name, ts.mean(), ts.std(), ts.max()))
        # Loose sanity bound only; the comparison is done by eye across the
        # two tables.
        self.assertLess(ts.mean(), 1 * ms)

    def test_none(self):
        self._bench("none", WL_NONE)

    def test_list(self):
        self._bench("list", WL_LIST)

    def test_object(self):
        self._bench("object", WL_OBJECT)

    def test_tuple_of_lists(self):
        self._bench("tuple_of_lists", WL_TUPLE_OF_LISTS)

    def test_list_of_lists(self):
        self._bench("list_of_lists", WL_LIST_OF_LISTS)

    def test_burst(self):
        self._bench("burst", WL_BURST)

    def test_range(self):
        self._bench("range", WL_RANGE)


class AllocTimingCell64RCTest(_AllocTimingCell64Mixin, ExperimentCase):
    """Traditional reference counting: eager malloc/free."""
    ctrc = False


class AllocTimingCell64CTRCTest(_AllocTimingCell64Mixin, ExperimentCase):
    """Same workloads inside ``with critical(...)``: slab allocation, deferred drop."""
    ctrc = True
