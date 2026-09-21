#!/usr/bin/env python3

import argparse
import os
import shutil
import re
import struct
from functools import partial
from collections import defaultdict
from datetime import datetime

from sipyco import common_args

from artiq import __version__ as artiq_version
from artiq.flashing import artifact_path, discover_bins
from artiq.remoting import SSHClient, LocalClient


def get_argparser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="ARTIQ flashing/deployment tool",
        epilog="""\
Valid commands:

    * write: write the binary or image file(s) to the specified flash region(s).
    * erase: erase the specified flash region(s).
    * load: load the main gateware bitstream into device (volatile but fast).
    * start: trigger the target to (re)load its gateware bitstream from flash.
      If your core device is reachable by network, prefer 'artiq_coremgmt reboot'. 
    * backup: read flash partitions and save them under a user-defined backup
      directory. Gateware, storage and firmware are truncated to their actual data
      size, while the complete bootloader partition is preserved.

Valid regions for write and erase actions:
    
    * gateware
    * bootloader
    * storage
    * firmware
    * Example: gateware,bootloader,storage,firmware

Prerequisites:

    * Connect the board through its/a JTAG adapter.
    * Have OpenOCD installed and in your $PATH.
    * Have access to the JTAG adapter's devices. Udev rules from OpenOCD:
      'sudo cp openocd/contrib/99-openocd.rules /etc/udev/rules.d'
      and replug the device. Ensure you are member of the
      plugdev group: 'sudo adduser $USER plugdev' and re-login.
""")

    parser.add_argument("--version", action="version",
                        version="ARTIQ v{}".format(artiq_version),
                        help="print the ARTIQ version number")

    common_args.verbosity_args(parser)

    parser.add_argument("-n", "--dry-run",
                        default=False, action="store_true",
                        help="only show the openocd script that would be run")
    parser.add_argument("-H", "--host", metavar="HOSTNAME",
                        type=str, default=None,
                        help="SSH host where the board is located")
    parser.add_argument("-J", "--jump",
                        type=str, default=None,
                        help="SSH host to jump through")
    parser.add_argument("-t", "--target", default="kasli",
                        help="target board, default: %(default)s, one of: "
                             "kasli phaser efc1v0 efc1v1 efc1v2 kc705")
    parser.add_argument("-I", "--preinit-command", default=[], action="append",
                        help="add a pre-initialization OpenOCD command. "
                             "Useful for selecting a board when several are connected.")
    parser.add_argument("-f", "--storage", help="write file to storage area")
    parser.add_argument("-d", "--dir", default=None, help="look for board binaries in this directory")
    parser.add_argument("--srcbuild", help="board binaries directory is laid out as a source build tree",
                        default=False, action="store_true")
    parser.add_argument("cmds", metavar="COMMANDS", nargs="*",
                        default=["write", "start"],
                        help="run cmd(s), for write and erase command, use format: CMD=REGION | "
                             "for other commands, use format: CMD | "
                             "default: flash gateware, firmware and bootloader and restart the FPGA device")
    return parser

def openocd_root():
    openocd = shutil.which("openocd")
    if not openocd:
        raise FileNotFoundError("OpenOCD is required but was not found in PATH. Is it installed?")
    return os.path.dirname(os.path.dirname(openocd))


def scripts_path():
    p = ["share", "openocd", "scripts"]
    if os.name == "nt":
        p.insert(0, "Library")
    return os.path.abspath(os.path.join(openocd_root(), *p))


def proxy_path():
    return os.path.abspath(os.path.join(openocd_root(), "share", "bscan-spi-bitstreams"))


def find_proxy_bitfile(filename):
    for p in [proxy_path(), os.path.expanduser("~/.migen"),
              "/usr/local/share/migen", "/usr/share/migen"]:
        full_path = os.path.join(p, filename)
        if os.access(full_path, os.R_OK):
            return full_path
    raise FileNotFoundError("Cannot find proxy bitstream {}"
                            .format(filename))


def add_commands(script, *commands, **substs):
    script += [command.format(**substs) for command in commands]


def get_full_partitions(config):
    result = {}
    previous_name, previous_bankname, previous_address = None, None, None

    for name, value in config.items():
        if name not in ["programmer", "flash_size"]:
            bankname, address = value
            if previous_name is not None:
                if address < previous_address:
                    raise ValueError(
                        "partition addresses are not in ascending order")
                result[previous_name] = (
                    previous_bankname,
                    previous_address,
                    address - previous_address
                )
            previous_name, previous_bankname, previous_address = name, bankname, address
    result[previous_name] = (
        previous_bankname,
        previous_address,
        config["flash_size"] - previous_address
    )
    return result


def get_fbi_size(filename):
    # An fbi starts with an 8-byte little-endian header containing two 32-bit
    # values: the firmware payload length and its CRC. The complete fbi size is
    # therefore the payload length plus the 8-byte header.
    with open(filename, "rb") as f:
        header = f.read(8)
    if len(header) != 8:
        raise ValueError("truncated FBI header")

    length, _ = struct.unpack("<II", header)
    return 8 + length


def get_storage_size(filename):
    # Storage consists of variable-length records. Each record starts with a
    # 32-bit big-endian value containing its total size. A value of 0xffffffff
    # marks the end of the stored records, include in the returned size
    with open(filename, "rb") as f:
        data = f.read()
    offset = 0
    while offset + 4 <= len(data):
        record_size = struct.unpack_from(">I", data, offset)[0]
        if record_size == 0xffffffff:
            return offset + 4
        if record_size < 4:
            raise ValueError("invalid storage record size")
        offset += record_size
    raise ValueError("storage terminator not found")


def get_gateware_size(filename):
    with open(filename, "rb") as f:
        data = f.read()

    # From UG470 (v1.17) table 6-1:
    # xilinx synchronization word
    sync = bytes.fromhex("aa995566")
    # DESYNC command
    desync = bytes.fromhex("300080010000000d")
    # xilinx NOOP packet
    noop = bytes.fromhex("20000000")
    # identifies the boundary between the original top.bin and
    # unused erased flash.
    erased = bytes.fromhex("ffffffff")

    sync_offset = data.find(sync)
    if sync_offset == -1:
        raise ValueError("Xilinx synchronization word not found in gateware")

    search_offset = sync_offset + len(sync)
    while True:
        desync_offset = data.find(desync, search_offset)
        if desync_offset == -1:
            raise ValueError(
                "Xilinx DESYNC command followed by erased flash "
                "not found in gateware"
            )
        offset = desync_offset + len(desync)
        while offset + 4 <= len(data) and data[offset:offset + 4] == noop:
            offset += 4
        if offset + 4 <= len(data) and data[offset:offset + 4] == erased:
            return offset
        search_offset = desync_offset + 4


def truncate_file(filename, size):
    with open(filename, "r+b") as f:
        f.truncate(size)


class Programmer:
    def __init__(self, client, preinit_script):
        self._client = client
        self._board_script = []
        self._preinit_script = [
            "gdb_port disabled",
            "tcl_port disabled",
            "telnet_port disabled"
        ] + preinit_script
        self._loaded = defaultdict(lambda: None)
        self._script = [
            "set error_msg \"Trying to use configured scan chain anyway\"",
            "if {[string first $error_msg [capture \"init\"]] != -1} {",
            "puts \"Found error and exiting\"",
            "exit}" 
        ]

    def _transfer_script(self, script):
        if isinstance(self._client, LocalClient):
            return "[find {}]".format(script)

        def rewriter(content):
            def repl(match):
                return self._transfer_script(match.group(1).decode()).encode()
            return re.sub(rb"\[find (.+?)\]", repl, content, flags=re.DOTALL)

        script = os.path.join(scripts_path(), script)
        return self._client.upload(script, rewriter)

    def add_flash_bank(self, name, tap, index):
        add_commands(self._board_script,
            "target create {tap}.{name}.proxy testee -chain-position {tap}.tap",
            "flash bank {name} jtagspi 0 0 0 0 {tap}.{name}.proxy {ir:#x}",
            tap=tap, name=name, ir=0x02 + index)

    def erase(self, target_regions, config, bankname="spi0"):
        self.load_proxy()

        firstsector, erase_list = None, []
        for region, t in config.items():
            if region in ["programmer", "flash_size"]:
                continue
            sector = t[1] // self._sector_size

            if firstsector is None and region in target_regions:
                firstsector = sector
            elif firstsector is not None and region not in target_regions:
                erase_list.append([firstsector, sector - 1])
                firstsector = None

        add_commands(self._script,"flash probe {bankname}", bankname=bankname)

        if firstsector is not None:
            erase_list.append([firstsector, "last"])

        for firstsector, lastsector in erase_list:
            add_commands(self._script,
                "flash erase_sector {bankname} {firstsector} {lastsector}",
                bankname=bankname, firstsector=firstsector, lastsector=lastsector)

    def load(self, bitfile, pld):
        os.stat(bitfile) # check for existence

        if self._loaded[pld] == bitfile:
            return
        self._loaded[pld] = bitfile

        bitfile = self._client.upload(bitfile)
        add_commands(self._script,
            "pld load {pld} {{{filename}}}",
            pld=pld, filename=bitfile)

    def load_proxy(self):
        raise NotImplementedError

    def write_binary(self, bankname, address, filename):
        self.load_proxy()

        size = os.path.getsize(filename)
        filename = self._client.upload(filename)
        add_commands(self._script,
            "flash probe {bankname}",
            "flash erase_sector {bankname} {firstsector} {lastsector}",
            "flash write_bank {bankname} {{{filename}}} {address:#x}",
            "flash verify_bank {bankname} {{{filename}}} {address:#x}",
            bankname=bankname, address=address, filename=filename,
            firstsector=address // self._sector_size,
            lastsector=(address + size - 1) // self._sector_size)

    def read_binary(self, bankname, address, length, filename):
        self.load_proxy()

        filename = self._client.prepare_download(filename)
        add_commands(self._script,
            "flash probe {bankname}",
            "flash read_bank {bankname} {{{filename}}} {address:#x} {length}",
            bankname=bankname, filename=filename, address=address, length=length)

    def start(self):
        raise NotImplementedError

    def script(self):
        return [
            *self._board_script,
            *self._preinit_script,
            *self._script,
            "exit"
        ]

    def run(self):
        cmdline = ["openocd"]
        if isinstance(self._client, LocalClient):
            cmdline += ["-s", scripts_path()]
        cmdline += ["-c", "; ".join(self.script())]

        cmdline = [arg.replace("{", "{{").replace("}", "}}") for arg in cmdline]
        self._client.run_command(cmdline)
        self._client.download()

        self._script = []


class ProgrammerXC7(Programmer):
    _sector_size = 0x10000

    def __init__(self, client, preinit_script, board, proxy):
        Programmer.__init__(self, client, preinit_script)
        self._proxy = proxy

        if board not in ["efc", "phaser"]:
            add_commands(self._board_script,
                "source {boardfile}",
                boardfile=self._transfer_script("board/{}.cfg".format(board)))
        else:
            add_commands(self._board_script,
                # OpenOCD does not have the efc board file so custom script is included.
                # To be used with Digilent-HS2 Programming Cable but the config in digilent-hs2.cfg is wrong
                # See digilent_jtag_smt2_nc.cfg for details
                "source [find interface/ftdi/digilent_jtag_smt2_nc.cfg]",

                "ftdi tdo_sample_edge falling",

                "reset_config none",
                "transport select jtag",
                "adapter speed 25000",

                "source [find cpld/xilinx-xc7.cfg]",
                "source [find cpld/jtagspi.cfg]",
                "source [find fpga/xilinx-xadc.cfg]",
                "source [find fpga/xilinx-dna.cfg]"
            )
        self.add_flash_bank("spi0", "xc7", index=0)

        add_commands(self._script, "xadc_report xc7.tap")

    def load_proxy(self):
        self.load(find_proxy_bitfile(self._proxy), pld=0)

    def start(self):
        add_commands(self._script,
            "xc7_program xc7.tap")


def main():
    args = get_argparser().parse_args()
    common_args.init_logger_from_args(args)

    config = {
        "kasli": {
            "programmer":   partial(ProgrammerXC7, board="kasli", proxy="bscan_spi_xc7a100t.bit"),
            "flash_size":            0x1000000,
            "gateware":     ("spi0", 0x000000),
            "bootloader":   ("spi0", 0x400000),
            "storage":      ("spi0", 0x440000),
            "firmware":     ("spi0", 0x450000),
        },
        "phaser": {
            "programmer":   partial(ProgrammerXC7, board="phaser", proxy="bscan_spi_xc7a100t.bit"),
            "flash_size":            0x1000000,
            "gateware":     ("spi0", 0x000000),
            "bootloader":   ("spi0", 0x400000),
            "storage":      ("spi0", 0x440000),
            "firmware":     ("spi0", 0x450000),
        },
        "efc1v0": {
            "programmer":   partial(ProgrammerXC7, board="efc", proxy="bscan_spi_xc7a100t.bit"),
            "flash_size":            0x1000000,
            "gateware":     ("spi0", 0x000000),
            "bootloader":   ("spi0", 0x600000),
            "storage":      ("spi0", 0x640000),
            "firmware":     ("spi0", 0x650000),
        },
        "efc1v1": {
            "programmer":   partial(ProgrammerXC7, board="efc", proxy="bscan_spi_xc7a200t.bit"),
            "flash_size":            0x1000000,
            "gateware":     ("spi0", 0x000000),
            "bootloader":   ("spi0", 0x600000),
            "storage":      ("spi0", 0x640000),
            "firmware":     ("spi0", 0x650000),
        },
        "efc1v2": {
            "programmer":   partial(ProgrammerXC7, board="efc", proxy="bscan_spi_xc7a200t.bit"),
            "flash_size":            0x1000000,
            "gateware":     ("spi0", 0x000000),
            "bootloader":   ("spi0", 0x600000),
            "storage":      ("spi0", 0x640000),
            "firmware":     ("spi0", 0x650000),
        },
        "kc705": {
            "programmer":   partial(ProgrammerXC7, board="kc705", proxy="bscan_spi_xc7k325t.bit"),
            "flash_size":            0x1000000, # From UG810 (v1.9)
            "gateware":     ("spi0", 0x000000),
            "bootloader":   ("spi0", 0xaf0000),
            "storage":      ("spi0", 0xb30000),
            "firmware":     ("spi0", 0xb40000),
        },
    }[args.target]

    cmds = []
    for cmd in args.cmds:
        cmd, *arguments = cmd.replace("=", ",").split(",")
        if cmd == "backup" and len(arguments) != 1:
            raise ValueError("the backup directory name must be provided")
        elif cmd in ["load", "start"] and arguments:
            raise ValueError(f"invalid command: {cmd}={','.join(arguments)}")
        elif cmd in ["write", "erase"]:
            if not arguments:
                if cmd == "write":
                    arguments = ["gateware", "bootloader", "firmware"]
                elif cmd == "erase":
                    arguments = ["gateware", "bootloader", "storage", "firmware"]
            if not(all(region in list(config)[2:] for region in arguments)):
                raise ValueError(f"unrecognized flash region(s): '{arguments}'")
        
        if cmd in ["write", "load"] and args.dir is None:
            if any(region in arguments for region in ["gateware", "bootloader", "firmware"]):
                raise ValueError("the directory containing the binaries needs to be specified using -d.")
        if cmd == "write" and args.storage is None:
            if "storage" in arguments:
                raise ValueError("the storage image file name needs to be specified using -f.")

        cmds.append([cmd, set(arguments)])

    binary_dir = args.dir

    if args.host is None:
        client = LocalClient()
    else:
        client = SSHClient(args.host, args.jump)

    programmer = config["programmer"](client, preinit_script=args.preinit_command)

    programmer_pending = False
    for cmd, arguments in cmds:
        if cmd != "backup":
            programmer_pending = True
        if cmd == "write":
            found_bins = (discover_bins(binary_dir, args.srcbuild)
                          if binary_dir is not None else {})
            for region in arguments:
                if region == "storage":
                    path = args.storage
                else:
                    try:
                        path = found_bins[region]
                    except KeyError:
                        raise FileNotFoundError(f"no binary found for {region}")
                programmer.write_binary(*config[region], path)
        elif cmd == "load":
            gateware_bin = artifact_path(binary_dir, "gateware", "top.bin")
            programmer.load(gateware_bin, 0)
        elif cmd == "start":
            programmer.start()
        elif cmd == "erase":
            programmer.erase(arguments, config)
        elif cmd == "backup":
            backup_directory = next(iter(arguments))
            backup_paths = {
                "gateware": artifact_path(
                    backup_directory,
                    "top.bin"
                ),
                "bootloader": artifact_path(
                    backup_directory,
                    "bootloader.bin"
                ),
                "storage": os.path.join(
                    backup_directory,
                    "storage.bin"
                ),
                "firmware": os.path.join(
                    backup_directory,
                    "firmware.fbi"
                )
            }
            for filename in backup_paths.values():
                os.makedirs(os.path.dirname(filename), exist_ok=True)

            for region, (bankname, address, partition_length) in get_full_partitions(config).items():
                programmer.read_binary(
                    bankname=bankname,
                    address=address,
                    length=partition_length,
                    filename=backup_paths[region]
                )

            if args.dry_run:
                print("\n".join(programmer.script()))
            else:
                programmer.run()
                truncate_file(
                    backup_paths["gateware"],
                    get_gateware_size(backup_paths["gateware"])
                )
                truncate_file(
                    backup_paths["storage"],
                    get_storage_size(backup_paths["storage"])
                )
                truncate_file(
                    backup_paths["firmware"],
                    get_fbi_size(backup_paths["firmware"])
                )
            programmer = config["programmer"](
                client,
                preinit_script=args.preinit_command
            )
        else:
            raise ValueError(f"invalid command: {cmd}")
    if programmer_pending:
        if args.dry_run:
            print("\n".join(programmer.script()))
        else:
            programmer.run()


if __name__ == "__main__":
    main()
