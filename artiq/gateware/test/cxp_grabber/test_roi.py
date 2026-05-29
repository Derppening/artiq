from itertools import combinations
import unittest

from migen import *
from misoc.cores.coaxpress.common import word_width, pixel_layout
from artiq.gateware.cxp_grabber.core import ROI, MonoPixelTracker
from artiq.gateware.test.cxp_grabber.common import get_frame, send_frame, set_roi_cfg


class DUT(Module):
    def __init__(self, pixel_layout, res_width, counts_width):
        self.pixel_source = Record(pixel_layout)

        self.submodules.tracker = tracker = MonoPixelTracker(
            self.pixel_source, res_width
        )
        self.submodules.roi_engine = ROI(tracker.tracked_pixels, counts_width)


class TestROI(unittest.TestCase):
    def test_run(self):
        supported_widths = [8, 10, 12, 14, 16]
        res_width = 16
        counts_width = 31

        # A handpicked frame sizes with width > max_pixel_per_cycle * 2
        # to at least fill the pixel_source fully twice
        height = 10
        width = 40

        for channel in [4, 2, 1]:
            max_pixel_per_cycle = (word_width * channel) // min(supported_widths)
            layout = pixel_layout(max_pixel_per_cycle, max(supported_widths))

            dut = DUT(layout, res_width, counts_width)
            fragment = dut.get_fragment()

            def receive(frame, x0, y0, x1, y1):
                # prepare expected outputs
                expected_pixels = []
                for line in frame[y0:y1]:
                    flatten_pixels = []
                    for buf in line:
                        flatten_pixels += buf
                    expected_pixels.extend(flatten_pixels[x0:x1])

                # verify outputs
                out = dut.roi_engine.out
                while (yield out.update) == 0:
                    yield

                count = yield out.count
                expected_count = sum(expected_pixels)
                self.assertEqual(expected_count, count)
                yield

            # The ROI is not affected by pixel_width, no need to test all supported width
            frame = get_frame(width, height, max(supported_widths), max_pixel_per_cycle)

            for x0, x1 in combinations(range(max_pixel_per_cycle), 2):
                y0, y1 = 0, height
                run_simulation(
                    fragment,
                    [
                        set_roi_cfg(x0, y0, x1, y1, dut.roi_engine.cfg),
                        send_frame(frame, dut.pixel_source),
                        receive(frame, x0, y0, x1, y1),
                    ],
                    clocks={"sys": 8},
                )


if __name__ == "__main__":
    unittest.main()
