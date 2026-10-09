"""Kernel heap footprint under traditional reference counting vs. CTRC.

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

Retained working sets are capped by the cell size: a list buffer of 24 ints or
24 pointers is 16 + 96 = 112 B, so ``retain_tree`` is 24 lists of 24 ints.

    python -m unittest -v artiq.test.coredevice.test_alloc_footprint
"""

from numpy import int32

from artiq.experiment import *
from artiq.coredevice.core import Core, heap_stats
from artiq.test.hardware_testbench import ExperimentCase


WL_CONTROL = 0
WL_RETAIN_FLAT = 1
WL_RETAIN_TREE = 2
WL_CHURN = 3
WL_EXCEPTIONS = 4
WL_RANGE_CHURN = 5

_CTRC_PAGES = 64          # 64 * 4 KiB reserved on entry, 1984 cells

_CHURN_ITERS = 10000
_EXCEPTION_ITERS = 256    # leaks 256 exception objects, < 1984 cells
_RANGE_ITERS = 1000


@compile
class _AllocFootprint(EnvExperiment):
    core: KernelInvariant[Core]
    heap_size: Kernel[int32]
    heap_size_end: Kernel[int32]
    busy_start: Kernel[int32]
    busy_entered: Kernel[int32]
    busy_live: Kernel[int32]
    busy_released: Kernel[int32]
    largest_free_released: Kernel[int32]

    def build(self):
        self.setattr_device("core")
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
            # 1 outer + 24 single-element lists, all kept alive
            return [[i] for i in range(24)]
        elif which == WL_RETAIN_TREE:
            # 1 outer + 24 lists of 24 ints (112 B buffers), all kept alive
            return [[i + j for j in range(24)] for i in range(24)]
        elif which == WL_CHURN:
            # allocate and drop 10000 small lists, keep nothing
            acc = int32(0)
            for i in range(_CHURN_ITERS):
                a = [i, i + 1]
                acc += a[1]
            return [[acc]]
        elif which == WL_EXCEPTIONS:
            # one exception object per raise, never released
            acc = int32(0)
            for i in range(_EXCEPTION_ITERS):
                try:
                    raise ValueError("footprint")
                except ValueError:
                    acc += 1
            return [[acc]]
        elif which == WL_RANGE_CHURN:
            # range objects (leaf; may be elided by LLVM above -O0)
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
        keep = self._workload(which)
        (busy, idle, largest_free) = heap_stats()
        self.busy_live = busy

    @kernel
    def _hold_ctrc(self, which: int32):
        """As ``_hold_rc``, with the workload inside ``with critical(...)``."""
        keep = [[int32(0)]]
        with critical(_CTRC_PAGES):
            (busy, idle, largest_free) = heap_stats()
            self.busy_entered = busy
            keep = self._workload(which)
        (busy, idle, largest_free) = heap_stats()
        self.busy_live = busy

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


class _AllocFootprintMixin:
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
        print("{} (ctrc={}, bytes relative to kernel start)".format(cls.__name__, cls.ctrc))
        print("| {} | busy at start | on entry | while live | after drop | largest free after drop |"
              .format("workload".ljust(width)))
        print("| {} | ------------- | -------- | ---------- | ---------- | ----------------------- |"
              .format("-" * width))
        for name, start, entered, live, released, largest in cls.results:
            print("| {} | {:>13d} | {:>8d} | {:>10d} | {:>10d} | {:>23d} |".format(
                name.ljust(width), start, entered, live, released, largest))

    def _measure(self, name, which):
        exp = self.create(_AllocFootprint)
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
                             exp.largest_free_released))

    def test_control(self):
        self._measure("control", WL_CONTROL)

    def test_retain_flat(self):
        self._measure("retain_flat", WL_RETAIN_FLAT)

    def test_retain_tree(self):
        self._measure("retain_tree", WL_RETAIN_TREE)

    def test_churn(self):
        self._measure("churn", WL_CHURN)

    def test_exceptions(self):
        self._measure("exceptions", WL_EXCEPTIONS)

    def test_range_churn(self):
        self._measure("range_churn", WL_RANGE_CHURN)


class AllocFootprintRCTest(_AllocFootprintMixin, ExperimentCase):
    """Traditional reference counting: eager malloc/free."""
    ctrc = False


class AllocFootprintCTRCTest(_AllocFootprintMixin, ExperimentCase):
    """Same workloads inside ``with critical(...)``: slab pages are reserved on
    entry and never returned."""
    ctrc = True
