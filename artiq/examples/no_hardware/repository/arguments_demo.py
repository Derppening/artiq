import logging

from artiq.experiment import *


class SubComponent1(HasEnvironment):
    def build(self):
        self.setattr_argument("sc1_scan",
            Scannable(default=[NoScan(3250), RangeScan(10, 20, 6, randomize=True)],
                      unit="kHz"),
            "Flux capacitor")
        self.setattr_argument("sc1_enum", EnumerationValue(["1", "2", "3"]),
                              "Flux capacitor")

    def do(self):
        print("SC1:")
        for i in self.sc1_scan:
            print(i)
        print(self.sc1_enum)


class SubComponent2(HasEnvironment):
    def build(self):
        self.setattr_argument("sc2_boolean", BooleanValue(False),
                              "Transporter")
        self.setattr_argument("sc2_scan", Scannable(
                                          default=RangeScan(200, 300, 49)),
                              "Transporter")
        self.setattr_argument("sc2_enum", EnumerationValue(["3", "4", "5"]),
                              "Transporter")

    def do(self):
        print("SC2:")
        print(self.sc2_boolean)
        for i in self.sc2_scan:
            print(i)
        print(self.sc2_enum)


class ArgumentsDemo(EnvExperiment):
    def build(self):
        # change the "foo" dataset and click the "recompute argument"
        # buttons.
        self.setattr_argument("pyon_value",
            PYONValue(self.get_dataset("foo", default=42)))
        self.setattr_argument("number", NumberValue(42e-6,
                                                    unit="us",
                                                    precision=4))
        self.setattr_argument("integer", NumberValue(42,
                                                     step=1, precision=0))
        self.setattr_argument("string", StringValue("Hello World"))
        self.setattr_argument("scan", Scannable(global_max=400,
                                                default=NoScan(325),
                                                precision=6))
        self.setattr_argument("boolean", BooleanValue(True), "Group")
        self.setattr_argument("enum", EnumerationValue(
            ["foo", "bar", "quux"], "foo"), "Group")

        row_headers = [str(x+1) for x in range(5)]
        column_headers = ["displacement", "velocity", "acceleration"]
        units = [["m","m/s", "m/s^2"] for row in range(5)]
        scale = [[1.0 for x in range(3)] for y in range(5)]
        content = [[1, 2, 2],[4, 4, 2],[9, 6, 2],[16, 8, 2],[25, 10, 2]]
        self.setattr_argument("table", TableValue(
            5, 3, row_headers, column_headers, content, unit=units, scale=scale))

        self.sc1 = SubComponent1(self)
        self.sc2 = SubComponent2(self)

    def run(self):
        logging.error("logging test: error")
        logging.warning("logging test: warning")
        logging.warning("logging test:" + " this is a very long message."*15)
        logging.info("logging test: info")
        logging.debug("logging test: debug")

        print(self.pyon_value)
        print(self.boolean)
        print(self.enum)
        print(self.number, type(self.number))
        print(self.integer, type(self.integer))
        print(self.string)
        for i in self.scan:
            print(i)
        for i in self.table:
            print(i)
        self.sc1.do()
        self.sc2.do()
