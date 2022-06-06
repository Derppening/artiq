import unittest

from numpy import int64

from artiq.experiment import *
from artiq.coredevice.comm_analyzer import (decode_dump, StoppedMessage,
                                            OutputMessage, InputMessage,
                                           _extract_log_chars, get_analyzer_dump)
from artiq.coredevice.core import Core
from artiq.coredevice.ttl import TTLOut, TTLInOut
from artiq.test.hardware_testbench import ExperimentCase


@nac3
class CreateTTLPulse(EnvExperiment):
    core: KernelInvariant[Core]
    loop_in: KernelInvariant[TTLInOut]
    loop_out: KernelInvariant[TTLOut]

    def build(self):
        self.setattr_device("core")
        self.setattr_device("loop_in")
        self.setattr_device("loop_out")

    @kernel
    def initialize_io(self):
        self.core.reset()
        self.loop_in.input()
        self.loop_out.output()
        self.core.delay(1.*us)
        self.loop_out.off()

    @kernel
    def run(self):
        self.core.break_realtime()
        with parallel:
            with sequential:
                delay_mu(int64(100))
                self.loop_out.pulse_mu(int64(1000))
            self.loop_in.count(self.loop_in.gate_both_mu(int64(1200)))


@nac3
class WriteLog(EnvExperiment):
    core: KernelInvariant[Core]

    def build(self):
        self.setattr_device("core")

    @kernel
    def run(self):
        self.core.reset()
        # NAC3TODO rtio_log("foo", 32)


class AnalyzerTest(ExperimentCase):
    @unittest.skip("NAC3TODO https://git.m-labs.hk/M-Labs/nac3/issues/298")
    def test_ttl_pulse(self):
        core_host = self.device_mgr.get_desc("core")["arguments"]["host"]

        exp = self.create(CreateTTLPulse)
        exp.initialize_io()
        get_analyzer_dump(core_host)  # clear analyzer buffer
        exp.run()

        dump = decode_dump(get_analyzer_dump(core_host))
        self.assertIsInstance(dump.messages[-1], StoppedMessage)
        output_messages = [msg for msg in dump.messages
                           if isinstance(msg, OutputMessage)
                              and msg.address == 0]
        self.assertEqual(len(output_messages), 2)
        self.assertEqual(
            abs(output_messages[0].timestamp - output_messages[1].timestamp),
            1000)
        input_messages = [msg for msg in dump.messages
                          if isinstance(msg, InputMessage)]
        self.assertEqual(len(input_messages), 2)
        # on Kasli systems, this has to go through the isolated DIO card
        self.assertAlmostEqual(
            abs(input_messages[0].timestamp - input_messages[1].timestamp),
            1000, delta=4)

    @unittest.skip("NAC3TODO https://git.m-labs.hk/M-Labs/nac3/issues/297")
    def test_rtio_log(self):
        core_host = self.device_mgr.get_desc("core")["arguments"]["host"]

        exp = self.create(WriteLog)
        get_analyzer_dump(core_host)  # clear analyzer buffer
        exp.run()

        dump = decode_dump(get_analyzer_dump(core_host))
        log  = "".join([_extract_log_chars(msg.data)
                        for msg in dump.messages
                        if isinstance(msg, OutputMessage) and msg.channel == dump.log_channel])
        self.assertEqual(log, "foo\x1E32\x1D")
