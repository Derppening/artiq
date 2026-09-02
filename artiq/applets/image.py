#!/usr/bin/env python3

import PyQt6  # make sure pyqtgraph imports Qt6
import pyqtgraph
import pyqtgraph.exporters

from artiq.applets.simple import SimpleApplet


class Image(pyqtgraph.ImageView):
    def __init__(self, args, req):
        pyqtgraph.ImageView.__init__(self)
        self.args = args
        copy_option = self.getImageItem().getViewBox().menu.addAction("Copy image to clipboard")
        def copy_image():
            exporter = pyqtgraph.exporters.ImageExporter(self.getImageItem())
            exporter.export(copy=True)
        copy_option.triggered.connect(copy_image)

    def data_changed(self, value, metadata, persist, mods):
        try:
            img = value[self.args.img]
        except KeyError:
            return
        self.setImage(img)


def main():
    applet = SimpleApplet(Image)
    applet.add_dataset("img", "image data (2D numpy array)")
    applet.run()

if __name__ == "__main__":
    main()
