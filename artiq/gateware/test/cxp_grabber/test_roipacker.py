from itertools import combinations
from multiprocessing import Pool
import unittest

from migen import *
from misoc.cores.coaxpress.common import word_width, pixel_layout
from artiq.gateware.cxp_grabber.core import ROIPacker, MonoPixelTracker
from artiq.gateware.test.cxp_grabber.common import get_frame, send_frame, set_roi_cfg


class DUT(Module):
    def __init__(self, pixel_layout, res_width, counts_width):
        self.pixel_source = Record(pixel_layout)

        self.submodules.tracker = tracker = MonoPixelTracker(
            self.pixel_source, res_width
        )
        self.submodules.roi_packer = ROIPacker(tracker.tracked_pixels)


def run(dut, fragment, frame, x0, y0, x1, y1, max_pixel_per_cycle, pixel_width):
    def receive(frame, x0, y0, x1, y1, max_pixel_per_cycle, max_pixel_width):

        # prepare expected outputs
        roi_flatten_pixels = []
        for line in frame[y0:y1]:
            flatten_pixels = []
            for buf in line:
                flatten_pixels += buf
            roi_flatten_pixels.extend(flatten_pixels[x0:x1])

        expected_data = []
        for offset in range(0, len(roi_flatten_pixels), max_pixel_per_cycle):
            data = 0
            for i, p in enumerate(
                roi_flatten_pixels[offset : offset + max_pixel_per_cycle]
            ):
                data |= p << (i * max_pixel_width)
            expected_data.append(data)

        # verify outputs
        source = dut.roi_packer.source
        for expected in expected_data:
            while (yield source.stb) == 0:
                yield
            data = yield source.data
            # Use assert instead of assertEqual from unittest
            # to workaround "TypeError: cannot pickle 'TextIOWrapper' instances"
            assert (
                expected == data
            ), f"Expected expected : {hex(expected)}, got : {hex(data)}"
            yield

    run_simulation(
        fragment,
        [
            set_roi_cfg(x0, y0, x1, y1, dut.roi_packer.cfg),
            send_frame(frame, dut.pixel_source),
            receive(
                frame,
                x0,
                y0,
                x1,
                y1,
                max_pixel_per_cycle,
                pixel_width,
            ),
        ],
        clocks={"sys": 8},
    )


class TestROIPacker(unittest.TestCase):
    def test_run(self):
        supported_widths = [8, 10, 12, 14, 16]
        res_width = 16
        counts_width = 31

        # A handpicked frame sizes with width > max_pixel_per_cycle * 2
        # to at least fill the pixel_source fully twice
        height = 10
        width = 40

        for ch in [4, 2, 1]:
            max_pixel_per_cycle = (word_width * ch) // min(supported_widths)

            layout = pixel_layout(max_pixel_per_cycle, max(supported_widths))

            dut = DUT(layout, res_width, counts_width)
            fragment = dut.get_fragment()

            # The ROI is not affected by pixel_width, no need to test all supported width
            frame = get_frame(width, height, max(supported_widths), max_pixel_per_cycle)

            # Parallelize the slow migen simulation
            with Pool(processes=10) as pool:
                pool.starmap(
                    run,
                    [
                        (
                            dut,
                            fragment,
                            frame,
                            x0,
                            0,
                            x1,
                            height,
                            max_pixel_per_cycle,
                            max(supported_widths),
                        )
                        for (x0, x1) in combinations(range(max_pixel_per_cycle * 2), 2)
                    ],
                )


if __name__ == "__main__":
    unittest.main()
