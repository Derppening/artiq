"""Allocation latency under traditional reference counting vs. CTRC.

Every workload allocates (and, on the next iteration, drops) a handful of heap
objects between two reads of the RTIO counter, so the per-iteration time is the
cost of the allocator plus the cost of the refcount release. The same kernels
run twice: once plainly (eager ``malloc``/``free`` through ksupport) and once
wrapped in ``with critical(...)`` (NAC3's CTRC slab, deferred drop).

Every object built here fits in one 128-byte CTRC cell on the 32-bit core
device: a list object is 16 B, a list buffer is 16 B + 4 B per ``int32``, so a
list of up to 28 ``int32`` fits, and a ``_Point`` is 16 B.

Two rows are deliberately degenerate:

* ``none`` has an empty timed region. It measures the two syscalls and the loop
  itself; subtract it from every other row.
* ``object`` and ``range`` build leaf objects, whose alloc/free pair LLVM may
  delete entirely at ``NAC3_OPT_LEVEL`` > 0. A near-zero delta there is the
  optimisation working, not the allocator.

Run both classes and compare the two tables printed at the end:

    python -m unittest -v artiq.test.coredevice.test_alloc_timing
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
WL_EXCEPTION = 7

# Pages reserved by ``with critical(...)``. One page is 31 cells. The exception
# workload leaks one cell per iteration (nothing releases an exception object),
# so the reservation must cover ``n`` iterations plus the live working set of
# the largest workload (``burst``: 1 outer + 24 inner lists, 2 cells each).
_CTRC_PAGES = 64


@compile
class _Point:
    x: Kernel[int32]
    y: Kernel[int32]

    @kernel
    def __init__(self, x: int32, y: int32):
        self.x = x
        self.y = y


@compile
class _AllocTiming(EnvExperiment):
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
                # 1 leaf object; alloc/free may be elided by LLVM
                p = _Point(i, i)
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
                # 1 outer (16 B + 24 ptr = 112 B buffer) + 24 inner lists,
                # plus one range object: ~51 cells live, all dropped at once
                b = [[i] for _ in range(24)]
                acc += b[23][0]
            elif which == WL_RANGE:
                # 1 leaf object; alloc/free may be elided by LLVM
                for j in range(i, i + 2):
                    acc += j
            elif which == WL_EXCEPTION:
                # 1 exception object per iteration, never released
                try:
                    raise ValueError("alloc timing")
                except ValueError:
                    acc += 1
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


class _AllocTimingMixin:
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
        print("{} (n={}, ctrc={})".format(cls.__name__, cls.n, cls.ctrc))
        print("| {} | mean (us) |  std (us) |  max (us) |".format("workload".ljust(width)))
        print("| {} | --------- | --------- | --------- |".format("-" * width))
        for name, mean, std, mx in cls.results:
            print("| {} | {:>9.3f} | {:>9.3f} | {:>9.3f} |".format(
                name.ljust(width), mean * 1e6, std * 1e6, mx * 1e6))

    def _bench(self, name, which):
        exp = self.create(_AllocTiming, n=self.n)
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

    def test_exception(self):
        self._bench("exception", WL_EXCEPTION)


class AllocTimingRCTest(_AllocTimingMixin, ExperimentCase):
    """Traditional reference counting: eager malloc/free."""
    ctrc = False


class AllocTimingCTRCTest(_AllocTimingMixin, ExperimentCase):
    """Same workloads inside ``with critical(...)``: slab allocation, deferred drop."""
    ctrc = True
