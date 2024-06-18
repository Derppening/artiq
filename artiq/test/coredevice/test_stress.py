import os
import time
import unittest
from numpy import int32

from artiq.experiment import *
from artiq.test.hardware_testbench import ExperimentCase


# NAC3TODO: bytes support
#@nac3
#class _Stress(EnvExperiment):
#    def build(self):
#        self.setattr_device("core")
#
#    @rpc # NAC3TODO: (flags={"async"})  
#    def sink(self, data: bytes):
#        pass
#
#    @kernel
#    def async_rpc(self, n: int32):
#        for _ in range(n):
#            self.sink(b"")


@unittest.skip("nac3 bytes")
class StressTest(ExperimentCase):
    def test_async_rpc(self):
        exp = self.create(_Stress)
        exp.async_rpc(16000)
