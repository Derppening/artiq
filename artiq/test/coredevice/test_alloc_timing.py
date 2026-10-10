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

The ``pattern_*`` rows instead build up a live set of objects and time each
allocation on its own, so the allocator works against a heap that already holds
up to ``n`` objects. The objects are ``_Obj32`` (32 B) and ``_Obj64`` (64 B),
both one CTRC cell, and are stored into holder lists that are allocated, filled
with one shared placeholder, before ``with critical(...)``: a holder is far
larger than a cell, and filling it with distinct objects would leave ``n`` extra
blocks on the RC heap.

* ``pattern_fill_32``, ``pattern_fill_64``: allocate ``n`` objects of one size.
* ``pattern_interleaved``: allocate ``n / 2`` pairs of a 32 B then a 64 B
  object. The reverse order is not run: neither allocator sees a difference.
* ``pattern_reuse_64_32``: allocate ``n`` 64 B objects, free every other one,
  then time allocating ``n / 2`` 32 B objects, which fit in the holes.
* ``pattern_reuse_32_64``: the same with the sizes swapped; the 64 B objects
  do not fit in the 32 B holes.

Only the allocations are timed: the setup of the reuse rows is not, and nothing
is dropped inside the timed region.

Run both classes and compare the two tables printed at the end:

    python -m unittest -v artiq.test.coredevice.test_alloc_timing
"""

import numpy
from numpy import int32, int64

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

PAT_FILL_32 = 0
PAT_FILL_64 = 1
PAT_INTERLEAVED = 2
PAT_REUSE_64_32 = 3
PAT_REUSE_32_64 = 4

# Pages reserved by ``with critical(...)``. One page is 31 cells. The exception
# workload leaks one cell per iteration (nothing releases an exception object),
# so the reservation must cover ``n`` iterations plus the live working set of
# the largest workload (``burst``: 1 outer + 24 inner lists, 2 cells each).
# The pattern workloads keep at most ``n`` objects live, one cell each.
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
class _Obj32:
    """8 B header + 3 ``int64`` = 32 B."""
    f0: Kernel[int64]
    f1: Kernel[int64]
    f2: Kernel[int64]

    @kernel
    def __init__(self, v: int32):
        self.f0 = int64(v)
        self.f1 = int64(v)
        self.f2 = int64(v)


@compile
class _Obj64:
    """8 B header + 7 ``int64`` = 64 B."""
    f0: Kernel[int64]
    f1: Kernel[int64]
    f2: Kernel[int64]
    f3: Kernel[int64]
    f4: Kernel[int64]
    f5: Kernel[int64]
    f6: Kernel[int64]

    @kernel
    def __init__(self, v: int32):
        self.f0 = int64(v)
        self.f1 = int64(v)
        self.f2 = int64(v)
        self.f3 = int64(v)
        self.f4 = int64(v)
        self.f5 = int64(v)
        self.f6 = int64(v)


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


@compile
class _AllocPattern(EnvExperiment):
    core: KernelInvariant[Core]
    n: KernelInvariant[int32]
    ts: Kernel[list[float]]

    def build(self, n=1000):
        self.setattr_device("core")
        self.n = n
        self.ts = [0.0] * n

    @kernel
    def _pattern(self, which: int32, p32: _Obj32, p64: _Obj64,
                 h32: list[_Obj32], h64: list[_Obj64],
                 r32: list[_Obj32], r64: list[_Obj64]):
        half = self.n // 2
        if which == PAT_FILL_32:
            for i in range(self.n):
                t0 = self.core.get_rtio_counter_mu()
                h32[i] = _Obj32(i)
                t1 = self.core.get_rtio_counter_mu()
                self.ts[i] = self.core.mu_to_seconds(t1 - t0)
        elif which == PAT_FILL_64:
            for i in range(self.n):
                t0 = self.core.get_rtio_counter_mu()
                h64[i] = _Obj64(i)
                t1 = self.core.get_rtio_counter_mu()
                self.ts[i] = self.core.mu_to_seconds(t1 - t0)
        elif which == PAT_INTERLEAVED:
            for i in range(half):
                t0 = self.core.get_rtio_counter_mu()
                h32[i] = _Obj32(i)
                t1 = self.core.get_rtio_counter_mu()
                self.ts[2 * i] = self.core.mu_to_seconds(t1 - t0)
                t0 = self.core.get_rtio_counter_mu()
                h64[i] = _Obj64(i)
                t1 = self.core.get_rtio_counter_mu()
                self.ts[2 * i + 1] = self.core.mu_to_seconds(t1 - t0)
        elif which == PAT_REUSE_64_32:
            for i in range(self.n):
                h64[i] = _Obj64(i)
            for i in range(half):
                h64[2 * i + 1] = p64
            for i in range(half):
                t0 = self.core.get_rtio_counter_mu()
                r32[i] = _Obj32(i)
                t1 = self.core.get_rtio_counter_mu()
                self.ts[i] = self.core.mu_to_seconds(t1 - t0)
        elif which == PAT_REUSE_32_64:
            for i in range(self.n):
                h32[i] = _Obj32(i)
            for i in range(half):
                h32[2 * i + 1] = p32
            for i in range(half):
                t0 = self.core.get_rtio_counter_mu()
                r64[i] = _Obj64(i)
                t1 = self.core.get_rtio_counter_mu()
                self.ts[i] = self.core.mu_to_seconds(t1 - t0)

    # As with ``_AllocTiming``, one entry point per mode. The holders are
    # allocated before ``with critical(...)``, since they do not fit in a cell.

    @kernel
    def pattern_rc(self, which: int32):
        p32 = _Obj32(0)
        p64 = _Obj64(0)
        h32 = [p32 for _ in range(self.n)]
        h64 = [p64 for _ in range(self.n)]
        r32 = [p32 for _ in range(self.n // 2)]
        r64 = [p64 for _ in range(self.n // 2)]
        self._pattern(which, p32, p64, h32, h64, r32, r64)

    @kernel
    def pattern_ctrc(self, which: int32):
        p32 = _Obj32(0)
        p64 = _Obj64(0)
        h32 = [p32 for _ in range(self.n)]
        h64 = [p64 for _ in range(self.n)]
        r32 = [p32 for _ in range(self.n // 2)]
        r64 = [p64 for _ in range(self.n // 2)]
        with critical(_CTRC_PAGES):
            self._pattern(which, p32, p64, h32, h64, r32, r64)


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
        self._record(name, ts)

    def _bench_pattern(self, name, which, count):
        exp = self.create(_AllocPattern, n=self.n)
        if self.ctrc:
            exp.pattern_ctrc(which)
        else:
            exp.pattern_rc(which)
        self._record(name, numpy.array(exp.ts[:count]))

    def _record(self, name, ts):
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

    def test_pattern_fill_32(self):
        self._bench_pattern("pattern_fill_32", PAT_FILL_32, self.n)

    def test_pattern_fill_64(self):
        self._bench_pattern("pattern_fill_64", PAT_FILL_64, self.n)

    def test_pattern_interleaved(self):
        self._bench_pattern("pattern_interleaved", PAT_INTERLEAVED, self.n)

    def test_pattern_reuse_64_32(self):
        self._bench_pattern("pattern_reuse_64_32", PAT_REUSE_64_32, self.n // 2)

    def test_pattern_reuse_32_64(self):
        self._bench_pattern("pattern_reuse_32_64", PAT_REUSE_32_64, self.n // 2)


class AllocTimingRCTest(_AllocTimingMixin, ExperimentCase):
    """Traditional reference counting: eager malloc/free."""
    ctrc = False


class AllocTimingCTRCTest(_AllocTimingMixin, ExperimentCase):
    """Same workloads inside ``with critical(...)``: slab allocation, deferred drop."""
    ctrc = True
