"""Kernel heap footprint under traditional reference counting vs. CTRC, sized
for 64-byte CTRC cells.

This is ``test_alloc_footprint.py`` adapted to a NAC3 built with
``CTRC_CELL_SIZE = 64`` (NAC3's ``experiment-cell64`` branch). Run it against
that build; the 128-byte file does not compile there. Differences from the
128-byte file:

* ``retain_flat`` and ``retain_tree`` keep 10 lists instead of 24, and
  ``retain_tree``'s lists hold 10 ints instead of 24, so every buffer is at most
  60 B (see below).
* There is no ``exceptions`` workload. An exception object is 72 B, which no
  longer fits in a cell, so ``raise`` inside ``with critical(...)`` is rejected.
  The RC leak per exception is unchanged from the 128-byte file.
* ``_CTRC_PAGES`` is 32: 32 * 63 = 2016 cells, at least the 1984 objects that
  64 pages held with 128-byte cells. A cell64 build also reserves 8 pages, not
  16, at kernel entry, which shows up in the CTRC class's ``busy at start``.

Kernels allocate from a heap owned by ksupport (``__kernel_heap_start`` ..
``__kernel_heap_end``, 64 MiB on the RISC-V targets) that is reset at every
kernel load. The ``heap_stats`` syscall reports its occupancy in bytes,
allocator block headers included, so every number below is what the workload
actually costs the heap, not what NAC3 requested.

Each test runs one kernel, ``measure_rc`` or ``measure_ctrc``, on a freshly
reset heap, sampling ``heap_stats`` four times:

1. ``start``: before anything else.
2. ``entered``: just inside ``with critical(...)`` for the CTRC class, just
   before the workload for the RC class. Under CTRC the difference from
   ``start`` is the slab pages reserved on entry, which are never returned.
3. ``live``: after the workload, with the objects it returns still alive.
4. ``released``: after ``_hold_rc``/``_hold_ctrc`` returns and drops them.

Between ``entered`` and ``live`` the heap's high-water marks are tracked with
``heap_peak_reset``/``heap_peak`` (firmware feature ``heap_peak``): the most
bytes busy at once, which catches objects allocated and freed inside the
workload, and the most live allocations at once. Under CTRC the slab pages are
one allocation each and the cells inside them are not seen; NAC3's ``memprof``
build reports those.

The two modes have separate kernel entry points. A kernel that may enter
``with critical`` reserves CTRC pages on entry, before ``start``, and NAC3
decides that from which functions are reachable, not from branch conditions, so
an entry point shared by both modes would make the RC kernel reserve too. The
RC class's ``start`` is therefore the heap with nothing on it.

``largest_free`` at ``released`` shows fragmentation: ksupport only joins
adjacent free blocks lazily inside ``malloc``.

Under CTRC, ``live`` and ``released`` read the same as ``entered`` unless the
workload outgrows the reservation, because slab cells come out of pages that
are already counted as busy. The slab's own occupancy is reported by NAC3's
``memprof`` build, not here.

Retained working sets are capped by the cell size. A list buffer is 16 B + 4 B
per slot, rounded up to a multiple of 20 B, so it fits in a 64-byte cell with at
most 11 slots; a comprehension over ``range(n)`` allocates ``n + 1`` slots, so
``retain_tree`` is 10 lists of 10 ints (16 + 11 * 4 = 60 B buffers).

    python -m unittest -v artiq.test.coredevice.test_alloc_footprint_cell64
"""

from numpy import int32

from artiq.experiment import *
from artiq.coredevice.core import Core, heap_stats, heap_peak, heap_peak_reset
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
    heap_size: Kernel[int32]
    heap_size_end: Kernel[int32]
    busy_start: Kernel[int32]
    busy_entered: Kernel[int32]
    busy_live: Kernel[int32]
    busy_released: Kernel[int32]
    largest_free_released: Kernel[int32]
    peak_bytes: Kernel[int32]
    peak_blocks: Kernel[int32]

    def build(self):
        self.setattr_device("core")
        self.heap_size = 0
        self.heap_size_end = 0
        self.busy_start = 0
        self.busy_entered = 0
        self.busy_live = 0
        self.busy_released = 0
        self.largest_free_released = 0
        self.peak_bytes = 0
        self.peak_blocks = 0

    # ---- workloads ---------------------------------------------------------

    @kernel
    def _workload(self, which: int32) -> list[list[int32]]:
        if which == WL_RETAIN_FLAT:
            # 1 outer + 10 single-element lists, all kept alive
            return [[i] for i in range(10)]
        elif which == WL_RETAIN_TREE:
            # 1 outer + 10 lists of 10 ints (60 B buffers), all kept alive
            return [[i + j for j in range(10)] for i in range(10)]
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
    def _sample_start(self):
        (busy, idle, largest_free) = heap_stats()
        self.heap_size = busy + idle
        self.busy_start = busy

    @kernel
    def _sample_released(self):
        (busy, idle, largest_free) = heap_stats()
        self.busy_released = busy
        self.largest_free_released = largest_free
        self.heap_size_end = busy + idle

    @kernel
    def _hold_rc(self, which: int32):
        """Run the workload and sample with its result alive. The result is
        dropped when this returns."""
        keep = [[int32(0)]]
        (busy, idle, largest_free) = heap_stats()
        self.busy_entered = busy
        heap_peak_reset()
        keep = self._workload(which)
        (busy, idle, largest_free) = heap_stats()
        self.busy_live = busy
        (peak_bytes, peak_blocks) = heap_peak()
        self.peak_bytes = peak_bytes
        self.peak_blocks = peak_blocks

    @kernel
    def _hold_ctrc(self, which: int32):
        """As ``_hold_rc``, with the workload inside ``with critical(...)``."""
        keep = [[int32(0)]]
        with critical(_CTRC_PAGES):
            (busy, idle, largest_free) = heap_stats()
            self.busy_entered = busy
            heap_peak_reset()
            keep = self._workload(which)
        (busy, idle, largest_free) = heap_stats()
        self.busy_live = busy
        (peak_bytes, peak_blocks) = heap_peak()
        self.peak_bytes = peak_bytes
        self.peak_blocks = peak_blocks

    # One entry point per mode. NAC3 makes a kernel reserve CTRC pages on entry
    # if a ``with critical`` block is reachable from it, whatever the branch
    # conditions on the way, so the RC kernel must not reach one at all.

    @kernel
    def measure_rc(self, which: int32):
        self._sample_start()
        self._hold_rc(which)
        self._sample_released()

    @kernel
    def measure_ctrc(self, which: int32):
        self._sample_start()
        self._hold_ctrc(which)
        self._sample_released()


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
        print("{} (ctrc={}, 64 B cells, bytes relative to kernel start, peak live blocks absolute)".format(cls.__name__, cls.ctrc))
        print("| {} | busy at start | on entry | while live | after drop | largest free after drop "
              "| peak during workload | peak live blocks |"
              .format("workload".ljust(width)))
        print("| {} | ------------- | -------- | ---------- | ---------- | ----------------------- "
              "| -------------------- | ---------------- |"
              .format("-" * width))
        for name, start, entered, live, released, largest, peak, blocks in cls.results:
            print("| {} | {:>13d} | {:>8d} | {:>10d} | {:>10d} | {:>23d} | {:>20d} | {:>16d} |".format(
                name.ljust(width), start, entered, live, released, largest, peak, blocks))

    def _measure(self, name, which):
        exp = self.create(_AllocFootprintCell64)
        if exp.core.target == "cortexa9":
            self.skipTest("heap_stats is not provided by the Zynq firmware")
        if self.ctrc:
            exp.measure_ctrc(which)
        else:
            exp.measure_rc(which)
        self.assertGreater(exp.heap_size, 0, "heap_stats reported an empty heap")
        self.assertEqual(exp.heap_size, exp.heap_size_end,
                         "busy + idle changed during the kernel")
        start = exp.busy_start
        self.results.append((name, start,
                             exp.busy_entered - start,
                             exp.busy_live - start,
                             exp.busy_released - start,
                             exp.largest_free_released,
                             exp.peak_bytes - start,
                             exp.peak_blocks))

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
