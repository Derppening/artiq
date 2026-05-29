from math import ceil

def get_frame(
    width, height, pixel_width, max_pixel_per_cycle, use_max_pixel_value=False
):
    frame = []
    for y in range(height):
        buffer = [[] for _ in range(ceil(width / max_pixel_per_cycle))]
        for x in range(width):
            max_pixel_value = (1 << pixel_width) - 1
            gray = (
                max_pixel_value
                if use_max_pixel_value
                else (x + y * width) & max_pixel_value
            )
            buffer[x // max_pixel_per_cycle].append(gray)
        frame.append(buffer)
    return frame

def send_frame(frame, pixel_source):
    for y, line in enumerate(frame):
        yield pixel_source.eol.eq(0)
        yield pixel_source.eof.eq(0)
        for x, buf in enumerate(line):
            yield pixel_source.px_stb.eq(((1 << len(buf)) - 1))
            for n, px in enumerate(buf):
                yield getattr(pixel_source, f"px{n}").eq(px)

            if x == len(line) - 1:
                yield pixel_source.eol.eq(1)
                if y == len(frame) - 1:
                    yield pixel_source.eof.eq(1)

            yield

    yield pixel_source.px_stb.eq(0)
    yield pixel_source.eol.eq(0)
    yield pixel_source.eof.eq(0)
    yield

def set_roi_cfg(x0, y0, x1, y1, cfg):
    yield cfg.x0.eq(x0)
    yield cfg.y0.eq(y0)
    yield cfg.x1.eq(x1)
    yield cfg.y1.eq(y1)
    yield
