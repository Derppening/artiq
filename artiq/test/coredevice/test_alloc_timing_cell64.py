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
* ``_CTRC_PAGES`` is 133, so the file also runs on the 128- and 256-byte cell
  builds: 133 * 15 = 1995 cells with 256-byte cells, at least the 1984 objects
  that 64 pages hold with 128-byte cells. With 64- and 128-byte cells it is
  133 * 63 = 8379 and 133 * 31 = 4123 cells.

The ``pattern_*`` rows build up a live set of objects and time each allocation
on its own, so the allocator works against a heap that already holds up to
``n`` objects. The objects are ``_Obj16Cell64`` (16 B) and ``_Obj32Cell64``
(32 B), one class smaller than in the 128-byte file, and are stored into holder
lists that are allocated, filled with one shared placeholder, before
``with critical(...)``: a holder is far larger than a cell, and filling it with
distinct objects would leave ``n`` extra blocks on the RC heap.

* ``pattern_fill_16``, ``pattern_fill_32``: allocate ``n`` objects of one size.
* ``pattern_interleaved``: allocate ``n / 2`` pairs of a 16 B then a 32 B
  object. The reverse order is not run: neither allocator sees a difference.
* ``pattern_reuse_32_16``: allocate ``n`` 32 B objects, free every other one,
  then time allocating ``n / 2`` 16 B objects, which fit in the holes.
* ``pattern_reuse_16_32``: the same with the sizes swapped; the 32 B objects
  do not fit in the 16 B holes.

Only the allocations are timed: the setup of the reuse rows is not, and nothing
is dropped inside the timed region.

``none`` has an empty timed region; subtract it from every other row.

    python -m unittest -v artiq.test.coredevice.test_alloc_timing_cell64
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

PAT_FILL_16 = 0
PAT_FILL_32 = 1
PAT_INTERLEAVED = 2
PAT_REUSE_32_16 = 3
PAT_REUSE_16_32 = 4

# Pages reserved by ``with critical(...)``. One page is 63, 31 or 15 cells
# with 64-, 128- or 256-byte cells. The largest workload of the ``_loop`` rows
# (``burst``) keeps 1 outer + 10 inner lists, 2 cells each, plus one range
# object live at a time. The pattern workloads keep at most ``n`` objects live,
# one cell each, which 133 pages hold even with 256-byte cells (1995 cells).
_CTRC_PAGES = 133


@compile
class _PointCell64:
    x: Kernel[int32]
    y: Kernel[int32]

    @kernel
    def __init__(self, x: int32, y: int32):
        self.x = x
        self.y = y


@compile
class _Obj16Cell64:
    """8 B header + 1 ``int64`` = 16 B."""
    f0: Kernel[int64]

    @kernel
    def __init__(self, v: int32):
        self.f0 = int64(v)


@compile
class _Obj32Cell64:
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


@compile
class _AllocPatternCell64(EnvExperiment):
    core: KernelInvariant[Core]
    n: KernelInvariant[int32]
    ts: Kernel[list[float]]

    def build(self, n=1000):
        self.setattr_device("core")
        self.n = n
        self.ts = [0.0] * n

    @kernel
    def _pattern(self, which: int32, p16: _Obj16Cell64, p32: _Obj32Cell64,
                 h16: list[_Obj16Cell64], h32: list[_Obj32Cell64],
                 r16: list[_Obj16Cell64], r32: list[_Obj32Cell64]):
        half = self.n // 2
        if which == PAT_FILL_16:
            for i in range(self.n):
                t0 = self.core.get_rtio_counter_mu()
                h16[i] = _Obj16Cell64(i)
                t1 = self.core.get_rtio_counter_mu()
                self.ts[i] = self.core.mu_to_seconds(t1 - t0)
        elif which == PAT_FILL_32:
            for i in range(self.n):
                t0 = self.core.get_rtio_counter_mu()
                h32[i] = _Obj32Cell64(i)
                t1 = self.core.get_rtio_counter_mu()
                self.ts[i] = self.core.mu_to_seconds(t1 - t0)
        elif which == PAT_INTERLEAVED:
            for i in range(half):
                t0 = self.core.get_rtio_counter_mu()
                h16[i] = _Obj16Cell64(i)
                t1 = self.core.get_rtio_counter_mu()
                self.ts[2 * i] = self.core.mu_to_seconds(t1 - t0)
                t0 = self.core.get_rtio_counter_mu()
                h32[i] = _Obj32Cell64(i)
                t1 = self.core.get_rtio_counter_mu()
                self.ts[2 * i + 1] = self.core.mu_to_seconds(t1 - t0)
        elif which == PAT_REUSE_32_16:
            for i in range(self.n):
                h32[i] = _Obj32Cell64(i)
            for i in range(half):
                h32[2 * i + 1] = p32
            for i in range(half):
                t0 = self.core.get_rtio_counter_mu()
                r16[i] = _Obj16Cell64(i)
                t1 = self.core.get_rtio_counter_mu()
                self.ts[i] = self.core.mu_to_seconds(t1 - t0)
        elif which == PAT_REUSE_16_32:
            for i in range(self.n):
                h16[i] = _Obj16Cell64(i)
            for i in range(half):
                h16[2 * i + 1] = p16
            for i in range(half):
                t0 = self.core.get_rtio_counter_mu()
                r32[i] = _Obj32Cell64(i)
                t1 = self.core.get_rtio_counter_mu()
                self.ts[i] = self.core.mu_to_seconds(t1 - t0)

    # As with ``_AllocTimingCell64``, one entry point per mode. The holders are
    # allocated before ``with critical(...)``, since they do not fit in a cell.

    @kernel
    def pattern_rc(self, which: int32):
        p16 = _Obj16Cell64(0)
        p32 = _Obj32Cell64(0)
        h16 = [p16 for _ in range(self.n)]
        h32 = [p32 for _ in range(self.n)]
        r16 = [p16 for _ in range(self.n // 2)]
        r32 = [p32 for _ in range(self.n // 2)]
        self._pattern(which, p16, p32, h16, h32, r16, r32)

    @kernel
    def pattern_ctrc(self, which: int32):
        p16 = _Obj16Cell64(0)
        p32 = _Obj32Cell64(0)
        h16 = [p16 for _ in range(self.n)]
        h32 = [p32 for _ in range(self.n)]
        r16 = [p16 for _ in range(self.n // 2)]
        r32 = [p32 for _ in range(self.n // 2)]
        with critical(_CTRC_PAGES):
            self._pattern(which, p16, p32, h16, h32, r16, r32)


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
        self._record(name, numpy.array(exp.ts))

    def _bench_pattern(self, name, which, count):
        exp = self.create(_AllocPatternCell64, n=self.n)
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

    def test_pattern_fill_16(self):
        self._bench_pattern("pattern_fill_16", PAT_FILL_16, self.n)

    def test_pattern_fill_32(self):
        self._bench_pattern("pattern_fill_32", PAT_FILL_32, self.n)

    def test_pattern_interleaved(self):
        self._bench_pattern("pattern_interleaved", PAT_INTERLEAVED, self.n)

    def test_pattern_reuse_32_16(self):
        self._bench_pattern("pattern_reuse_32_16", PAT_REUSE_32_16, self.n // 2)

    def test_pattern_reuse_16_32(self):
        self._bench_pattern("pattern_reuse_16_32", PAT_REUSE_16_32, self.n // 2)


class AllocTimingCell64RCTest(_AllocTimingCell64Mixin, ExperimentCase):
    """Traditional reference counting: eager malloc/free."""
    ctrc = False


class AllocTimingCell64CTRCTest(_AllocTimingCell64Mixin, ExperimentCase):
    """Same workloads inside ``with critical(...)``: slab allocation, deferred drop."""
    ctrc = True
