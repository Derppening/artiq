import unittest

from migen import *
from misoc.cores.coaxpress.common import word_width, pixel_layout
from artiq.gateware.cxp_grabber.core import MonoPixelTracker
from artiq.gateware.test.cxp_grabber.common import get_frame, send_frame


class DUT(Module):
    def __init__(self, pixel_layout, res_width):
        self.pixel_source = Record(pixel_layout)
        self.submodules.tracker = MonoPixelTracker(self.pixel_source, res_width)


class TestMonoPixelTracker(unittest.TestCase):
    def test_run(self):
        supported_widths = [8, 10, 12, 14, 16]
        res_width = 16

        height = 1
        width = 8 * 8
        for channel in [4, 2, 1]:
            max_pixel_per_cycle = (word_width * channel) // min(supported_widths)
            layout = pixel_layout(max_pixel_per_cycle, max(supported_widths))

            dut = DUT(layout, res_width)
            fragment = dut.get_fragment()

            def receive(frame):
                sources = dut.tracker.tracked_pixels
                for expected_y, line in enumerate(frame):
                    expected_x = 0
                    for x, buf in enumerate(line):
                        while (yield sources[0].stb) == 0:
                            yield

                        expected_eof = 0
                        if x == len(line) - 1 and expected_y == len(frame) - 1:
                            expected_eof = 1

                        for n, expected_gray in enumerate(buf):
                            px = sources[n]
                            rx_eof = yield px.eof
                            rx_gray = yield px.gray
                            rx_stb = yield px.stb
                            rx_x = yield px.x
                            rx_y = yield px.y

                            self.assertEqual(rx_eof, expected_eof)
                            self.assertEqual(rx_gray, expected_gray)
                            self.assertEqual(rx_stb, 1)
                            self.assertEqual(rx_x, expected_x)
                            self.assertEqual(rx_y, expected_y)

                            expected_x = expected_x + 1

                        yield

            for pixel_width in supported_widths:
                frame = get_frame(
                    width, height, max(supported_widths), max_pixel_per_cycle
                )

                run_simulation(
                    fragment,
                    [
                        send_frame(frame, dut.pixel_source),
                        receive(frame),
                    ],
                    clocks={"sys": 8},
                )


if __name__ == "__main__":
    unittest.main()
