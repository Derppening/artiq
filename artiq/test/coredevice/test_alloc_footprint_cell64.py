"""Kernel heap footprint under traditional reference counting vs. CTRC, sized
for 64-byte CTRC cells.

This is ``test_alloc_footprint.py`` adapted to a NAC3 built with
``CTRC_CELL_SIZE = 64`` (NAC3's ``experiment-cell64`` branch). Run it against
that build; the 128-byte file does not compile there. Differences from the
128-byte file:

* ``retain_flat`` and ``retain_tree`` keep 12 lists instead of 24, and
  ``retain_tree``'s lists hold 12 ints instead of 24, so every buffer is at most
  16 + 12 * 4 = 64 B.
* There is no ``exceptions`` workload. An exception object is 72 B, which no
  longer fits in a cell, so ``raise`` inside ``with critical(...)`` is rejected.
  The RC leak per exception is unchanged from the 128-byte file.
* ``_CTRC_PAGES`` is 32: 32 * 63 = 2016 cells, at least the 1984 objects that
  64 pages held with 128-byte cells. A cell64 build also reserves 8 pages, not
  16, at kernel entry, which shows up in ``busy at start``.

Kernels allocate from a heap owned by ksupport (``__kernel_heap_start`` ..
``__kernel_heap_end``, 64 MiB on the RISC-V targets) that is reset at every
kernel load. The ``heap_stats`` syscall reports its occupancy in bytes,
allocator block headers included, so every number below is what the workload
actually costs the heap, not what NAC3 requested.

Each test runs one kernel, ``measure``, on a freshly reset heap, sampling
``heap_stats`` four times:

1. ``start``: before anything else.
2. ``entered``: just inside ``with critical(...)`` for the CTRC class, just
   before the workload for the RC class. Under CTRC the difference from
   ``start`` is the slab pages reserved on entry, which are never returned.
3. ``live``: after the workload, with the objects it returns still alive.
4. ``released``: after ``_hold`` returns and drops them.

``largest_free`` at ``released`` shows fragmentation: ksupport only joins
adjacent free blocks lazily inside ``malloc``.

Under CTRC, ``live`` and ``released`` read the same as ``entered`` unless the
workload outgrows the reservation, because slab cells come out of pages that
are already counted as busy. The slab's own occupancy is reported by NAC3's
``memprof`` build, not here.

Retained working sets are capped by the cell size: a list buffer of 12 ints or
12 pointers is 16 + 48 = 64 B, so ``retain_tree`` is 12 lists of 12 ints.

    python -m unittest -v artiq.test.coredevice.test_alloc_footprint_cell64
"""

from numpy import int32

from artiq.experiment import *
from artiq.coredevice.core import Core, heap_stats
from artiq.test.hardware_testbench import ExperimentCase


WL_CONTROL = 0
WL_RETAIN_FLAT = 1
WL_RETAIN_TREE = 2
WL_CHURN = 3
WL_RANGE_CHURN = 5

_CTRC_PAGES = 32          # 32 * 4 KiB reserved on entry, 2016 cells

_CHURN_ITERS = 10000
_RANGE_ITERS = 1000


@compile
class _AllocFootprintCell64(EnvExperiment):
    core: KernelInvariant[Core]
    ctrc: KernelInvariant[bool]
    heap_size: Kernel[int32]
    heap_size_end: Kernel[int32]
    busy_start: Kernel[int32]
    busy_entered: Kernel[int32]
    busy_live: Kernel[int32]
    busy_released: Kernel[int32]
    largest_free_released: Kernel[int32]

    def build(self, ctrc=False):
        self.setattr_device("core")
        self.ctrc = ctrc
        self.heap_size = 0
        self.heap_size_end = 0
        self.busy_start = 0
        self.busy_entered = 0
        self.busy_live = 0
        self.busy_released = 0
        self.largest_free_released = 0

    # ---- workloads ---------------------------------------------------------

    @kernel
    def _workload(self, which: int32) -> list[list[int32]]:
        if which == WL_RETAIN_FLAT:
            # 1 outer + 12 single-element lists, all kept alive
            return [[i] for i in range(12)]
        elif which == WL_RETAIN_TREE:
            # 1 outer + 12 lists of 12 ints (64 B buffers), all kept alive
            return [[i + j for j in range(12)] for i in range(12)]
        elif which == WL_CHURN:
            # allocate and drop 10000 small lists, keep nothing
            acc = int32(0)
            for i in range(_CHURN_ITERS):
                a = [i, i + 1]
                acc += a[1]
            return [[acc]]
        elif which == WL_RANGE_CHURN:
            # range objects (leaf)
            acc = int32(0)
            for i in range(_RANGE_ITERS):
                for j in range(i, i + 2):
                    acc += j
            return [[acc]]
        return [[int32(0)]]

    # ---- measurement -------------------------------------------------------

    @kernel
    def _hold(self, which: int32):
        """Run the workload and sample with its result alive. The result is
        dropped when this returns."""
        keep = [[int32(0)]]
        if self.ctrc:
            with critical(_CTRC_PAGES):
                (busy, idle, largest_free) = heap_stats()
                self.busy_entered = busy
                keep = self._workload(which)
        else:
            (busy, idle, largest_free) = heap_stats()
            self.busy_entered = busy
            keep = self._workload(which)
        (busy, idle, largest_free) = heap_stats()
        self.busy_live = busy

    @kernel
    def measure(self, which: int32):
        (busy, idle, largest_free) = heap_stats()
        self.heap_size = busy + idle
        self.busy_start = busy
        self._hold(which)
        (busy, idle, largest_free) = heap_stats()
        self.busy_released = busy
        self.largest_free_released = largest_free
        self.heap_size_end = busy + idle


class _AllocFootprintCell64Mixin:
    """Test bodies shared by the RC and CTRC classes. Not a TestCase itself."""

    ctrc = False

    @classmethod
    def setUpClass(cls):
        cls.results = []

    @classmethod
    def tearDownClass(cls):
        if not cls.results:
            return
        width = max(len(r[0]) for r in cls.results)
        print()
        print("{} (ctrc={}, 64 B cells, bytes relative to kernel start)".format(cls.__name__, cls.ctrc))
        print("| {} | busy at start | on entry | while live | after drop | largest free after drop |"
              .format("workload".ljust(width)))
        print("| {} | ------------- | -------- | ---------- | ---------- | ----------------------- |"
              .format("-" * width))
        for name, start, entered, live, released, largest in cls.results:
            print("| {} | {:>13d} | {:>8d} | {:>10d} | {:>10d} | {:>23d} |".format(
                name.ljust(width), start, entered, live, released, largest))

    def _measure(self, name, which):
        exp = self.create(_AllocFootprintCell64, ctrc=self.ctrc)
        if exp.core.target == "cortexa9":
            self.skipTest("heap_stats is not provided by the Zynq firmware")
        exp.measure(which)
        self.assertGreater(exp.heap_size, 0, "heap_stats reported an empty heap")
        self.assertEqual(exp.heap_size, exp.heap_size_end,
                         "busy + idle changed during the kernel")
        start = exp.busy_start
        self.results.append((name, start,
                             exp.busy_entered - start,
                             exp.busy_live - start,
                             exp.busy_released - start,
                             exp.largest_free_released))

    def test_control(self):
        self._measure("control", WL_CONTROL)

    def test_retain_flat(self):
        self._measure("retain_flat", WL_RETAIN_FLAT)

    def test_retain_tree(self):
        self._measure("retain_tree", WL_RETAIN_TREE)

    def test_churn(self):
        self._measure("churn", WL_CHURN)

    def test_range_churn(self):
        self._measure("range_churn", WL_RANGE_CHURN)


class AllocFootprintCell64RCTest(_AllocFootprintCell64Mixin, ExperimentCase):
    """Traditional reference counting: eager malloc/free."""
    ctrc = False


class AllocFootprintCell64CTRCTest(_AllocFootprintCell64Mixin, ExperimentCase):
    """Same workloads inside ``with critical(...)``: slab pages are reserved on
    entry and never returned."""
    ctrc = True
