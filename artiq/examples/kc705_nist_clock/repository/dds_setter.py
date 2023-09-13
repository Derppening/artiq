from operator import itemgetter

from artiq.experiment import *
from artiq.coredevice.core import Core
from artiq.coredevice.ad9914 import AD9914


@nac3
class DDSSetter(EnvExperiment):
    """DDS Setter"""

    core: KernelInvariant[Core]

    def build(self):
        self.setattr_device("core")

        self.dds = dict()

        device_db = self.get_device_db()
        for k, v in sorted(device_db.items(), key=itemgetter(0)):
            if (isinstance(v, dict)
                    and v["type"] == "local"
                    and v["module"] == "artiq.coredevice.ad9914"
                    and v["class"] == "AD9914"):
                self.dds[k] = {
                    "driver": self.get_device(k),
                    "frequency": self.get_argument(
                        "{}_frequency".format(k),
                        NumberValue(100e6, scale=1e6, unit="MHz", precision=6))
                }

    @kernel
    def set_dds(self, dds: AD9914, frequency: float):
        self.core.break_realtime()
        dds.set(frequency)
        self.core.delay(200.*ms)

    def run(self):
        for k, v in self.dds.items():
            self.set_dds(v["driver"], v["frequency"])
